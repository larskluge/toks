"""Network-free unit tests for toks.

The tool is an extensionless executable, so load it as a module by path.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest import mock

_PATH = pathlib.Path(__file__).resolve().parent / "toks"
_loader = importlib.machinery.SourceFileLoader("toks", str(_PATH))
_spec = importlib.util.spec_from_loader("toks", _loader)
toks = importlib.util.module_from_spec(_spec)
_loader.exec_module(toks)


# ---- LM Studio listing -----------------------------------------------------

# Anonymized capture of GET /api/v0/models
LMSTUDIO_V0 = {
    "object": "list",
    "data": [
        {
            "id": "vision-scribe-7b",
            "object": "model",
            "type": "vlm",
            "publisher": "example-community",
            "arch": "qwen2_vl",
            "compatibility_type": "mlx",
            "quantization": "4bit",
            "state": "not-loaded",
            "max_context_length": 32768,
        },
        {
            "id": "tiny-embedder",
            "object": "model",
            "type": "embeddings",
            "publisher": "example-labs",
            "arch": "bert",
            "compatibility_type": "gguf",
            "quantization": "F16",
            "state": "not-loaded",
            "max_context_length": 512,
        },
        {
            "id": "router-moe-15b",
            "object": "model",
            "type": "llm",
            "publisher": "example-ai",
            "arch": "mixtral_moe",
            "compatibility_type": "gguf",
            "quantization": "Q4_K_M",
            "state": "loaded",
            "max_context_length": 32768,
            "loaded_context_length": 8192,
        },
    ],
}


class LMStudioListingTests(unittest.TestCase):
    def test_parses_all_models_into_records(self):
        records = toks.lmstudio_parse_models(LMSTUDIO_V0)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r.provider == "lmstudio" for r in records))

    def test_maps_id_format_quant_context(self):
        rec = toks.lmstudio_parse_models(LMSTUDIO_V0)[0]
        self.assertEqual(rec.name, "vision-scribe-7b")
        self.assertEqual(rec.fmt, "mlx")
        self.assertEqual(rec.quant, "4bit")
        self.assertEqual(rec.ctx_max, 32768)

    def test_loaded_context_length_captured(self):
        moe = toks.lmstudio_parse_models(LMSTUDIO_V0)[2]
        self.assertEqual(moe.ctx_loaded, 8192)

    def test_llm_and_vlm_are_benchmarkable_embeddings_are_not(self):
        recs = {r.name: r for r in toks.lmstudio_parse_models(LMSTUDIO_V0)}
        self.assertTrue(recs["vision-scribe-7b"].benchmarkable)   # vlm
        self.assertTrue(recs["router-moe-15b"].benchmarkable)     # llm
        self.assertFalse(recs["tiny-embedder"].benchmarkable)     # embeddings

    def test_moe_detected_from_arch(self):
        recs = {r.name: r for r in toks.lmstudio_parse_models(LMSTUDIO_V0)}
        self.assertTrue(recs["router-moe-15b"].moe)
        self.assertFalse(recs["vision-scribe-7b"].moe)

    def test_size_and_params_unknown_without_enrichment(self):
        rec = toks.lmstudio_parse_models(LMSTUDIO_V0)[0]
        self.assertIsNone(rec.size_bytes)
        self.assertEqual(rec.params, "-")

    def test_empty_or_missing_data_yields_no_records(self):
        self.assertEqual(toks.lmstudio_parse_models({}), [])
        self.assertEqual(toks.lmstudio_parse_models({"data": []}), [])


# ---- LM Studio v1 enrichment ----------------------------------------------

# Anonymized capture of the richer GET /api/v1/models (0.4.0+).
# NB: the real payload nests under "models" (not "data") and keys each entry by
# "key" (not "id"), with quantization as an object -- captured live, not from docs.
LMSTUDIO_V1 = {
    "object": "list",
    "models": [
        {
            "key": "vision-scribe-7b",
            "display_name": "Vision Scribe 7B",
            "architecture": "qwen2_vl",
            "quantization": {"name": "4bit", "bits_per_weight": 4},
            "size_bytes": 4_300_000_000,
            "params_string": "7B",
            "format": "mlx",
        },
        {
            "key": "router-moe-15b",
            "display_name": "Router MoE 15B",
            "architecture": "mixtral_moe",
            "quantization": {"name": "Q4_K_M", "bits_per_weight": 4},
            "size_bytes": 9_100_000_000,
            "params_string": "15B",
            "format": "gguf",
        },
    ],
}


class LMStudioEnrichmentTests(unittest.TestCase):
    def test_backfills_size_and_params_by_id(self):
        records = toks.lmstudio_parse_models(LMSTUDIO_V0)
        toks.lmstudio_enrich(records, LMSTUDIO_V1)
        by_name = {r.name: r for r in records}
        self.assertEqual(by_name["vision-scribe-7b"].size_bytes, 4_300_000_000)
        self.assertEqual(by_name["vision-scribe-7b"].params, "7B")

    def test_models_absent_from_v1_keep_dashes(self):
        records = toks.lmstudio_parse_models(LMSTUDIO_V0)
        toks.lmstudio_enrich(records, LMSTUDIO_V1)
        by_name = {r.name: r for r in records}
        self.assertIsNone(by_name["tiny-embedder"].size_bytes)
        self.assertEqual(by_name["tiny-embedder"].params, "-")

    def test_missing_v1_payload_is_a_noop(self):
        records = toks.lmstudio_parse_models(LMSTUDIO_V0)
        toks.lmstudio_enrich(records, None)        # must not raise
        self.assertIsNone(records[0].size_bytes)


# ---- benchmark stats parsing ----------------------------------------------


class LMStudioBenchTests(unittest.TestCase):
    def test_reads_tps_and_ttft_from_stats(self):
        resp = {
            "stats": {
                "tokens_per_second": 51.43,
                "time_to_first_token": 0.111,
                "generation_time": 0.954,
                "stop_reason": "eosFound",
            }
        }
        result = toks.parse_lmstudio_bench(resp)
        self.assertAlmostEqual(result.tokens_per_second, 51.43)
        self.assertAlmostEqual(result.time_to_first_token, 0.111)

    def test_missing_stats_returns_none(self):
        self.assertIsNone(toks.parse_lmstudio_bench({"choices": []}))

    def test_zero_tps_returns_none(self):
        self.assertIsNone(toks.parse_lmstudio_bench({"stats": {"tokens_per_second": 0}}))


class OllamaBenchTests(unittest.TestCase):
    def test_computes_tps_from_eval_counts(self):
        resp = {"eval_count": 128, "eval_duration": 1_000_000_000}  # 1s
        result = toks.parse_ollama_bench(resp)
        self.assertAlmostEqual(result.tokens_per_second, 128.0)

    def test_ttft_from_load_plus_prompt_eval(self):
        resp = {
            "eval_count": 100,
            "eval_duration": 1_000_000_000,
            "load_duration": 200_000_000,        # 0.2s
            "prompt_eval_duration": 300_000_000,  # 0.3s
        }
        result = toks.parse_ollama_bench(resp)
        self.assertAlmostEqual(result.time_to_first_token, 0.5)

    def test_missing_metrics_returns_none(self):
        self.assertIsNone(toks.parse_ollama_bench({}))
        self.assertIsNone(toks.parse_ollama_bench({"eval_count": 0, "eval_duration": 0}))


# ---- Ollama listing --------------------------------------------------------

# Anonymized /api/tags entries
OLLAMA_TAGS = [
    {
        "name": "scribe:8b",
        "digest": "aaaa1111",
        "size": 4_700_000_000,
        "modified_at": "2026-05-28T10:00:00Z",
        "details": {"format": "gguf", "quantization_level": "Q4_K_M", "family": "llama"},
    },
    {
        "name": "vision-mlx:26b-mlx",
        "digest": "bbbb2222",
        "size": 16_000_000_000,
        "modified_at": "2026-05-20T10:00:00Z",
        "details": {"format": "safetensors", "quantization_level": "4bit", "family": ""},
    },
]
# Anonymized /api/show model_info per model name
OLLAMA_INFO = {
    "scribe:8b": {
        "general.architecture": "llama",
        "general.parameter_count": 8_000_000_000,
        "llama.context_length": 40960,
    },
    "vision-mlx:26b-mlx": {
        "general.architecture": "qwen3_5_moe",
        "general.parameter_count": 26_000_000_000,
        "qwen3_5_moe.expert_count": 8,
        "qwen3_5_moe.expert_used_count": 2,
        "qwen3_5_moe.context_length": 32768,
    },
}


class OllamaListingTests(unittest.TestCase):
    def test_builds_records_with_core_fields(self):
        recs = {r.name: r for r in toks.ollama_parse_models(OLLAMA_TAGS, OLLAMA_INFO)}
        s = recs["scribe:8b"]
        self.assertEqual(s.provider, "ollama")
        self.assertEqual(s.digest, "aaaa1111")
        self.assertEqual(s.fmt, "gguf")
        self.assertEqual(s.quant, "Q4_K_M")
        self.assertEqual(s.size_bytes, 4_700_000_000)
        self.assertEqual(s.modified_at, "2026-05-28T10:00:00Z")
        self.assertEqual(s.ctx_max, 40960)
        self.assertTrue(s.benchmarkable)

    def test_safetensors_is_benchmarkable_and_moe_detected(self):
        recs = {r.name: r for r in toks.ollama_parse_models(OLLAMA_TAGS, OLLAMA_INFO)}
        v = recs["vision-mlx:26b-mlx"]
        self.assertTrue(v.benchmarkable)        # safetensors
        self.assertTrue(v.moe)                  # arch ends in _moe / expert_count>1
        self.assertEqual(v.ctx_max, 32768)

    def test_params_string_populated(self):
        recs = {r.name: r for r in toks.ollama_parse_models(OLLAMA_TAGS, OLLAMA_INFO)}
        self.assertEqual(recs["scribe:8b"].params, "8B")
        self.assertIn("MoE", recs["vision-mlx:26b-mlx"].params)


# ---- mlx listing -----------------------------------------------------------

# Anonymized capture of mlx_lm.server GET /v1/models (ids only).
MLX_MODELS = {
    "object": "list",
    "data": [
        {"id": "example-org/chat-mini-4bit", "object": "model", "created": 1780000000},
        {"id": "example-org/router-moe-8bit", "object": "model", "created": 1780000000},
        {"id": "example-org/image-gen-turbo", "object": "model", "created": 1780000000},
    ],
}


class MlxListingTests(unittest.TestCase):
    def test_parses_ids_into_records(self):
        records = toks.mlx_parse_models(MLX_MODELS)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r.provider == "mlx" for r in records))
        self.assertEqual(records[0].name, "example-org/chat-mini-4bit")

    def test_records_default_to_mlx_format_and_benchmarkable(self):
        rec = toks.mlx_parse_models(MLX_MODELS)[0]
        self.assertEqual(rec.fmt, "mlx")
        self.assertTrue(rec.benchmarkable)  # remote hosts can't check configs

    def test_metadata_unknown_without_enrichment(self):
        rec = toks.mlx_parse_models(MLX_MODELS)[0]
        self.assertIsNone(rec.size_bytes)
        self.assertEqual(rec.params, "-")
        self.assertEqual(rec.quant, "-")
        self.assertIsNone(rec.ctx_max)

    def test_empty_or_missing_data_yields_no_records(self):
        self.assertEqual(toks.mlx_parse_models({}), [])
        self.assertEqual(toks.mlx_parse_models({"data": []}), [])


# ---- mlx HF-cache enrichment -------------------------------------------------


def _make_hub_repo(hub, repo_dir, config, weights=b"w" * 9000):
    """Build a minimal HF-cache repo: refs/main -> snapshot with blob symlink."""
    repo = hub / repo_dir
    (repo / "refs").mkdir(parents=True)
    (repo / "blobs").mkdir()
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")
    blob = repo / "blobs" / "deadbeef"
    blob.write_bytes(weights)
    (snapshot / "model.safetensors").symlink_to(blob)
    if config is not None:
        (snapshot / "config.json").write_text(json.dumps(config))
    return snapshot


class MlxEnrichmentTests(unittest.TestCase):
    DENSE_CONFIG = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "max_position_embeddings": 8192,
        "quantization": {"group_size": 64, "bits": 4},
        "vocab_size": 32000,
    }
    MOE_CONFIG = {
        "model_type": "qwen3_next",
        "architectures": ["Qwen3NextForCausalLM"],
        "max_position_embeddings": 262144,
        "quantization": {"group_size": 64, "bits": 8},
        "num_experts": 8,
        "num_experts_per_tok": 2,
    }

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.hub = pathlib.Path(tmp.name)
        self.dense_snapshot = _make_hub_repo(
            self.hub, "models--example-org--chat-mini-4bit", self.DENSE_CONFIG)
        _make_hub_repo(self.hub, "models--example-org--router-moe-8bit",
                       self.MOE_CONFIG)
        # Diffusers-style repo: config.json only in a subdirectory.
        image_snapshot = _make_hub_repo(
            self.hub, "models--example-org--image-gen-turbo", config=None)
        (image_snapshot / "text_encoder").mkdir()
        (image_snapshot / "text_encoder" / "config.json").write_text("{}")
        self.records = toks.mlx_parse_models(MLX_MODELS)
        toks.mlx_enrich(self.records, self.hub)
        self.by_name = {r.name.split("/")[-1]: r for r in self.records}

    def test_size_summed_with_symlinks_resolved(self):
        rec = self.by_name["chat-mini-4bit"]
        expected = 9000 + len(json.dumps(self.DENSE_CONFIG))
        self.assertEqual(rec.size_bytes, expected)

    def test_quant_and_ctx_from_config(self):
        rec = self.by_name["chat-mini-4bit"]
        self.assertEqual(rec.quant, "4bit")
        self.assertEqual(rec.ctx_max, 8192)

    def test_params_estimated_from_size_and_bits(self):
        self.assertTrue(self.by_name["chat-mini-4bit"].params.startswith("~"))

    def test_moe_detected_with_expert_ratio(self):
        rec = self.by_name["router-moe-8bit"]
        self.assertTrue(rec.moe)
        self.assertIn("MoE 2/8", rec.params)
        self.assertFalse(self.by_name["chat-mini-4bit"].moe)

    def test_missing_top_level_config_demotes_benchmarkable(self):
        rec = self.by_name["image-gen-turbo"]
        self.assertFalse(rec.benchmarkable)
        self.assertEqual(rec.quant, "-")

    def test_modified_at_set_from_refs_mtime(self):
        rec = self.by_name["chat-mini-4bit"]
        self.assertIsNotNone(rec.modified_at)
        self.assertIsNotNone(toks.parse_modified(rec.modified_at))

    def test_unknown_repo_left_untouched(self):
        records = [toks.ModelRecord("mlx", "example-org/not-downloaded",
                                    benchmarkable=True)]
        toks.mlx_enrich(records, self.hub)
        self.assertIsNone(records[0].size_bytes)
        self.assertTrue(records[0].benchmarkable)

    def test_absolute_path_id_enriched_directly(self):
        records = [toks.ModelRecord("mlx", str(self.dense_snapshot),
                                    benchmarkable=True)]
        toks.mlx_enrich(records, self.hub)
        self.assertEqual(records[0].ctx_max, 8192)

    def test_missing_hub_is_a_noop(self):
        records = toks.mlx_parse_models(MLX_MODELS)
        toks.mlx_enrich(records, self.hub / "nonexistent")  # must not raise
        self.assertIsNone(records[0].size_bytes)


# ---- mlx streaming benchmark parse ------------------------------------------


class FakeClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


def _sse(*events):
    return [f"data: {json.dumps(e)}\n".encode() for e in events] + [b"data: [DONE]\n"]


class MlxBenchTests(unittest.TestCase):
    CHUNKS = [
        {"choices": [{"index": 0, "text": "Hello"}]},
        {"choices": [{"index": 0, "text": " world"}]},
        {"choices": [{"index": 0, "text": "!"}]},
    ]
    USAGE = {"choices": [],
             "usage": {"prompt_tokens": 2, "completion_tokens": 12,
                       "total_tokens": 14}}

    def test_tps_from_usage_tokens_and_chunk_times(self):
        lines = _sse(*self.CHUNKS, self.USAGE)
        result = toks.bench_from_sse(lines, start=0.5,
                                     clock=FakeClock([1.0, 2.0, 3.0]))
        # 12 tokens streamed between t=1.0 and t=3.0 -> (12-1)/2.0
        self.assertAlmostEqual(result.tokens_per_second, 5.5)
        self.assertAlmostEqual(result.time_to_first_token, 0.5)

    def test_falls_back_to_chunk_count_without_usage(self):
        lines = _sse(*self.CHUNKS)
        result = toks.bench_from_sse(lines, start=0.0,
                                     clock=FakeClock([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(result.tokens_per_second, 1.0)  # (3-1)/2.0

    def test_no_content_chunks_returns_none(self):
        self.assertIsNone(toks.bench_from_sse(_sse(self.USAGE), start=0.0,
                                              clock=FakeClock([])))

    def test_single_chunk_has_no_measurable_rate(self):
        lines = _sse(self.CHUNKS[0], self.USAGE)
        self.assertIsNone(toks.bench_from_sse(lines, start=0.0,
                                              clock=FakeClock([1.0])))

    def test_malformed_lines_skipped(self):
        lines = [b"data: not-json\n", b": comment\n", b"\n"] + _sse(
            *self.CHUNKS, self.USAGE)
        result = toks.bench_from_sse(lines, start=0.0,
                                     clock=FakeClock([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(result.tokens_per_second, 5.5)


# ---- mlx provider wiring -----------------------------------------------------


class MlxWiringTests(unittest.TestCase):
    def test_local_host_detection(self):
        self.assertTrue(toks.is_local_host("http://localhost:8080"))
        self.assertTrue(toks.is_local_host("http://127.0.0.1:8080"))
        self.assertFalse(toks.is_local_host("http://box.example.net:8080"))

    def test_select_providers_mlx(self):
        providers = toks.select_providers("mlx")
        self.assertEqual([p.name for p in providers], ["mlx"])

    def test_select_providers_all_includes_mlx(self):
        providers = toks.select_providers("all")
        self.assertEqual([p.name for p in providers],
                         ["ollama", "lmstudio", "mlx", "unsloth"])

    def test_provider_flag_accepts_mlx(self):
        self.assertEqual(toks.parse_args(["--provider", "mlx"]).provider, "mlx")

    def test_mlx_cache_key_and_migration_idempotent(self):
        rec = toks.ModelRecord("mlx", "example-org/chat-mini-4bit")
        self.assertEqual(toks.cache_key(rec), "mlx:example-org/chat-mini-4bit")
        cache = {toks.cache_key(rec): {"provider": "mlx", "tokens_per_second": 9.0}}
        self.assertEqual(toks.migrate_cache(cache), cache)


# ---- shared rendering helpers ---------------------------------------------


class TagCellTests(unittest.TestCase):
    def test_safetensors_renders_as_mlx(self):
        rec = toks.ModelRecord(provider="ollama", name="x", fmt="safetensors")
        self.assertEqual(toks.tag_cell(rec), "mlx")

    def test_gguf_and_mlx_passthrough(self):
        self.assertEqual(toks.tag_cell(toks.ModelRecord("o", "x", fmt="gguf")), "gguf")
        self.assertEqual(toks.tag_cell(toks.ModelRecord("l", "x", fmt="mlx")), "mlx")

    def test_moe_appended(self):
        rec = toks.ModelRecord("l", "x", fmt="gguf", moe=True)
        self.assertEqual(toks.tag_cell(rec), "gguf moe")

    def test_unknown_format_is_dash(self):
        self.assertEqual(toks.tag_cell(toks.ModelRecord("l", "x", fmt="")), "-")


class FormattingHelperTests(unittest.TestCase):
    def test_human_size_dash_for_none(self):
        self.assertEqual(toks.human_size(None), "-")

    def test_human_size_gigabytes(self):
        self.assertEqual(toks.human_size(4_700_000_000), "4.4 GB")

    def test_quant_bits_per_weight_q4(self):
        self.assertEqual(toks.quant_bits_per_weight("Q4_K_M"), 4.5)

    def test_human_ctx_dash_for_none_and_zero(self):
        self.assertEqual(toks.human_ctx(None), "-")
        self.assertEqual(toks.human_ctx(0), "-")

    def test_human_ctx_kilo(self):
        self.assertEqual(toks.human_ctx(131072), "128k")

    def test_human_ctx_rolls_over_to_mega(self):
        self.assertEqual(toks.human_ctx(1048576), "1M")       # 1024k -> 1M
        self.assertEqual(toks.human_ctx(1572864), "1.5M")
        self.assertEqual(toks.human_ctx(10485760), "10M")     # not 10240k


# ---- cache key + migration + lookup ---------------------------------------


class CacheKeyTests(unittest.TestCase):
    def test_ollama_key_uses_digest(self):
        rec = toks.ModelRecord("ollama", "scribe:8b", digest="aaaa1111")
        self.assertEqual(toks.cache_key(rec), "ollama:aaaa1111")

    def test_ollama_key_falls_back_to_name(self):
        rec = toks.ModelRecord("ollama", "scribe:8b", digest=None)
        self.assertEqual(toks.cache_key(rec), "ollama:scribe:8b")

    def test_lmstudio_key_uses_id(self):
        rec = toks.ModelRecord("lmstudio", "router-moe-15b")
        self.assertEqual(toks.cache_key(rec), "lmstudio:router-moe-15b")


class CacheMigrationTests(unittest.TestCase):
    def test_legacy_digest_entry_reprefixed_to_ollama(self):
        legacy = {"aaaa1111": {"model": "scribe:8b", "tokens_per_second": 100.0}}
        migrated = toks.migrate_cache(legacy)
        self.assertIn("ollama:aaaa1111", migrated)
        self.assertEqual(migrated["ollama:aaaa1111"]["provider"], "ollama")
        self.assertAlmostEqual(migrated["ollama:aaaa1111"]["tokens_per_second"], 100.0)

    def test_legacy_name_with_colon_not_mistaken_for_migrated(self):
        # An Ollama tag contains a colon; it must still be treated as legacy.
        legacy = {"scribe:8b": {"model": "scribe:8b", "tokens_per_second": 50.0}}
        migrated = toks.migrate_cache(legacy)
        self.assertIn("ollama:scribe:8b", migrated)

    def test_already_migrated_entries_unchanged_and_idempotent(self):
        current = {"lmstudio:router-moe-15b": {"provider": "lmstudio", "tokens_per_second": 9.0}}
        once = toks.migrate_cache(current)
        twice = toks.migrate_cache(once)
        self.assertEqual(once, current)
        self.assertEqual(twice, current)

    def test_migration_preserves_existing_ollama_benchmark_value(self):
        legacy = {"bbbb2222": {"model": "vision-mlx:26b-mlx", "tokens_per_second": 121.6}}
        rec = toks.ModelRecord("ollama", "vision-mlx:26b-mlx", digest="bbbb2222")
        migrated = toks.migrate_cache(legacy)
        self.assertAlmostEqual(toks.cached_tps(migrated, rec), 121.6)


class CacheLookupAndSortTests(unittest.TestCase):
    def setUp(self):
        self.cache = {
            "ollama:aaaa1111": {"tokens_per_second": 120.0, "time_to_first_token": 0.2},
            "lmstudio:router-moe-15b": {"tokens_per_second": 188.0},
        }

    def test_cached_tps_and_ttft(self):
        rec = toks.ModelRecord("ollama", "scribe:8b", digest="aaaa1111")
        self.assertAlmostEqual(toks.cached_tps(self.cache, rec), 120.0)
        self.assertAlmostEqual(toks.cached_ttft(self.cache, rec), 0.2)

    def test_uncached_returns_none(self):
        rec = toks.ModelRecord("lmstudio", "unknown-model")
        self.assertIsNone(toks.cached_tps(self.cache, rec))

    def test_sort_merges_providers_by_tps_unbenched_last(self):
        recs = [
            toks.ModelRecord("ollama", "scribe:8b", digest="aaaa1111"),       # 120
            toks.ModelRecord("lmstudio", "router-moe-15b"),                    # 188
            toks.ModelRecord("lmstudio", "cold-model"),                        # none
        ]
        ordered = sorted(recs, key=lambda r: toks.tps_sort_key(self.cache, r))
        self.assertEqual([r.name for r in ordered],
                         ["router-moe-15b", "scribe:8b", "cold-model"])


# ---- table rendering -------------------------------------------------------


class BuildRowsTests(unittest.TestCase):
    def setUp(self):
        self.cache = {"lmstudio:router-moe-15b": {"tokens_per_second": 188.0,
                                                  "time_to_first_token": 0.11}}
        self.records = [
            toks.ModelRecord("lmstudio", "router-moe-15b", fmt="gguf", moe=True,
                             params="15B", quant="Q4_K_M", ctx_max=32768,
                             ctx_loaded=8192, size_bytes=None),
            toks.ModelRecord("ollama", "scribe:8b", fmt="gguf", params="8B",
                             quant="Q4_K_M", ctx_max=40960, size_bytes=4_700_000_000,
                             modified_at="2026-05-28T10:00:00Z", digest="aaaa1111"),
        ]

    def test_header_leads_with_provider(self):
        rows = toks.build_rows(self.records, self.cache)
        self.assertEqual(rows[0][0], "PROVIDER")
        self.assertIn("TTFT", rows[0])
        self.assertIn("CTX", rows[0])

    def test_lmstudio_row_cells(self):
        rows = toks.build_rows(self.records, self.cache)
        header = rows[0]
        row = rows[1]  # router-moe-15b
        cell = dict(zip(header, row))
        self.assertEqual(cell["PROVIDER"], "lmstudio")
        self.assertEqual(cell["NAME"], "router-moe-15b")
        self.assertEqual(cell["TAG"], "gguf moe")
        self.assertEqual(cell["SIZE"], "-")            # unknown for LM Studio v0
        self.assertEqual(cell["TOKENS/S"], "188.0")
        self.assertEqual(cell["TTFT"], "0.11")
        self.assertEqual(cell["CTX"], "8k")            # loaded ctx preferred
        self.assertEqual(cell["MODIFIED"], "-")

    def test_ollama_row_uncached_shows_dashes(self):
        rows = toks.build_rows(self.records, self.cache)
        cell = dict(zip(rows[0], rows[2]))             # scribe:8b, not cached
        self.assertEqual(cell["TOKENS/S"], "-")
        self.assertEqual(cell["TTFT"], "-")
        self.assertEqual(cell["CTX"], "40k")           # max ctx fallback
        self.assertEqual(cell["SIZE"], "4.4 GB")

    def test_table_renders_aligned_string(self):
        rows = toks.build_rows(self.records, self.cache)
        out = toks.table(rows)
        self.assertIn("PROVIDER", out.splitlines()[0])
        self.assertEqual(len(out.splitlines()), 3)


# ---- CLI parsing + benchmark target selection ------------------------------


class ParseArgsTests(unittest.TestCase):
    def _error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                toks.parse_args(argv)

    def test_no_flags_means_no_bench(self):
        self.assertIsNone(toks.parse_args([]).bench)

    def test_bare_bench_is_empty_target_list(self):
        self.assertEqual(toks.parse_args(["--bench"]).bench, [])

    def test_bench_accepts_missing_and_all_keywords(self):
        self.assertEqual(toks.parse_args(["--bench", "missing"]).bench, ["missing"])
        self.assertEqual(toks.parse_args(["--bench", "all"]).bench, ["all"])

    def test_bench_accepts_model_names(self):
        args = toks.parse_args(["--bench", "scribe:8b", "router-moe-15b"])
        self.assertEqual(args.bench, ["scribe:8b", "router-moe-15b"])

    def test_keywords_cannot_combine_with_other_targets(self):
        self._error(["--bench", "all", "scribe:8b"])
        self._error(["--bench", "missing", "scribe:8b"])
        self._error(["--bench", "all", "missing"])

    def test_removed_flags_are_rejected(self):
        self._error(["--all"])
        self._error(["--missing"])
        self._error(["--model", "scribe:8b"])
        self._error(["--prompt", "hi"])
        self._error(["--max-tokens", "5"])

    def test_provider_flag_survives(self):
        self.assertEqual(toks.parse_args(["--provider", "ollama"]).provider, "ollama")


class SelectBenchRecordsTests(unittest.TestCase):
    def setUp(self):
        self.cache = {"ollama:aaaa1111": {"tokens_per_second": 120.0}}
        self.cached = toks.ModelRecord("ollama", "scribe:8b", digest="aaaa1111")
        self.cold = toks.ModelRecord("lmstudio", "router-moe-15b")
        self.records = [self.cached, self.cold]

    def test_empty_targets_selects_uncached_only(self):
        selected, unknown = toks.select_bench_records(self.records, [], self.cache)
        self.assertEqual(selected, [self.cold])
        self.assertEqual(unknown, [])

    def test_missing_keyword_same_as_empty(self):
        selected, _ = toks.select_bench_records(self.records, ["missing"], self.cache)
        self.assertEqual(selected, [self.cold])

    def test_all_selects_everything(self):
        selected, _ = toks.select_bench_records(self.records, ["all"], self.cache)
        self.assertEqual(selected, self.records)

    def test_names_select_exact_models_even_if_cached(self):
        selected, unknown = toks.select_bench_records(
            self.records, ["scribe:8b"], self.cache)
        self.assertEqual(selected, [self.cached])
        self.assertEqual(unknown, [])

    def test_unknown_names_reported(self):
        selected, unknown = toks.select_bench_records(
            self.records, ["nope", "scribe:8b"], self.cache)
        self.assertEqual(selected, [self.cached])
        self.assertEqual(unknown, ["nope"])


class RunBenchmarksTests(unittest.TestCase):
    class FakeProvider:
        def __init__(self):
            self.calls = []

        def benchmark(self, rec, prompt, max_tokens):
            self.calls.append((rec.name, prompt, max_tokens))
            return toks.BenchResult(42.0, 0.1)

    def test_benchmarks_with_default_prompt_and_tokens(self):
        rec = toks.ModelRecord("ollama", "scribe:8b", digest="aaaa1111",
                               benchmarkable=True)
        provider = self.FakeProvider()
        cache = {}
        with contextlib.redirect_stderr(io.StringIO()):
            changed = toks.run_benchmarks([rec], {"ollama": provider}, cache)
        self.assertTrue(changed)
        self.assertEqual(provider.calls,
                         [("scribe:8b", toks.DEFAULT_PROMPT, toks.DEFAULT_MAX_TOKENS)])
        entry = cache["ollama:aaaa1111"]
        self.assertAlmostEqual(entry["tokens_per_second"], 42.0)
        self.assertEqual(entry["max_tokens"], toks.DEFAULT_MAX_TOKENS)

    def test_non_benchmarkable_is_skipped(self):
        rec = toks.ModelRecord("lmstudio", "tiny-embedder", benchmarkable=False)
        provider = self.FakeProvider()
        cache = {}
        with contextlib.redirect_stderr(io.StringIO()):
            changed = toks.run_benchmarks([rec], {"lmstudio": provider}, cache)
        self.assertFalse(changed)
        self.assertEqual(provider.calls, [])


class HostNormalizationTests(unittest.TestCase):
    def test_adds_scheme_when_missing(self):
        self.assertEqual(toks.normalize_host("box.example.net:1234", "x"),
                         "http://box.example.net:1234")

    def test_strips_trailing_slash(self):
        self.assertEqual(toks.normalize_host("http://h:1234/", "x"), "http://h:1234")

    def test_keeps_https(self):
        self.assertEqual(toks.normalize_host("https://h:443", "x"), "https://h:443")

    def test_uses_default_when_empty(self):
        self.assertEqual(toks.normalize_host(None, "http://127.0.0.1:11434"),
                         "http://127.0.0.1:11434")


class MainDisplayTests(unittest.TestCase):
    """--bench restricts the printed table to the benchmarked models."""

    def _records(self):
        fast = toks.ModelRecord("ollama", "fast-model", digest="d1")
        cold = toks.ModelRecord("lmstudio", "cold-model")
        return [fast, cold]

    def _run_main(self, argv):
        records = self._records()
        cache = {"ollama:d1": {"tokens_per_second": 100.0}}
        # Records default to benchmarkable=False, so run_benchmarks skips them;
        # active only needs to be non-empty to clear the reachability guard.
        active = {"ollama": object(), "lmstudio": object()}
        out = io.StringIO()
        with mock.patch.object(toks, "gather_records", return_value=(records, active)), \
                mock.patch.object(toks, "load_cache", return_value=cache), \
                mock.patch.object(toks, "save_cache"), \
                mock.patch.object(sys, "argv", ["toks", *argv]), \
                contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            toks.main()
        return out.getvalue()

    def test_no_bench_shows_all_rows(self):
        output = self._run_main([])
        self.assertIn("fast-model", output)
        self.assertIn("cold-model", output)

    def test_bench_missing_shows_only_benched_rows(self):
        output = self._run_main(["--bench"])
        self.assertIn("cold-model", output)
        self.assertNotIn("fast-model", output)

    def test_bench_all_shows_all_rows(self):
        output = self._run_main(["--bench", "all"])
        self.assertIn("fast-model", output)
        self.assertIn("cold-model", output)

    def test_bench_named_shows_only_named_row(self):
        output = self._run_main(["--bench", "fast-model"])
        self.assertIn("fast-model", output)
        self.assertNotIn("cold-model", output)


# ---- bits/weight column ----------------------------------------------------


class BpwTests(unittest.TestCase):
    def test_effective_bpw_exact(self):
        rec = toks.ModelRecord("unsloth", "m", size_bytes=18_323_731_456,
                               param_count=30_697_345_596)
        self.assertAlmostEqual(toks.effective_bpw(rec), 4.776, places=2)

    def test_human_bpw_two_decimals(self):
        rec = toks.ModelRecord("ollama", "m", size_bytes=20_230_000_000,
                               param_count=31_273_088_876)
        self.assertEqual(toks.human_bpw(rec), "5.18")

    def test_human_bpw_tilde_when_estimated(self):
        rec = toks.ModelRecord("mlx", "m", size_bytes=1_000_000_000,
                               param_count=2_000_000_000, param_estimated=True)
        self.assertEqual(toks.human_bpw(rec), "~4.00")

    def test_dash_when_size_or_count_missing(self):
        self.assertIsNone(toks.effective_bpw(toks.ModelRecord("ollama", "m")))
        self.assertEqual(toks.human_bpw(toks.ModelRecord("ollama", "m")), "-")
        self.assertEqual(
            toks.human_bpw(toks.ModelRecord("ollama", "m", size_bytes=100)), "-")

    def test_header_has_bpw_after_params(self):
        self.assertEqual(toks.HEADER.index("BPW"), toks.HEADER.index("PARAMS") + 1)
        self.assertIn("BPW", toks.RIGHT_ALIGN)

    def test_row_renders_bpw_value(self):
        rec = toks.ModelRecord("unsloth", "demo-GGUF", fmt="gguf", params="31B",
                               size_bytes=18_323_731_456,
                               param_count=30_697_345_596)
        rows = toks.build_rows([rec], {})
        cell = dict(zip(rows[0], rows[1]))
        self.assertEqual(cell["BPW"], "4.78")

    def test_row_bpw_dash_when_unknown(self):
        rows = toks.build_rows([toks.ModelRecord("ollama", "x", params="8B")], {})
        cell = dict(zip(rows[0], rows[1]))
        self.assertEqual(cell["BPW"], "-")


# ---- ollama param-count split ----------------------------------------------


class OllamaParamCountTests(unittest.TestCase):
    def test_real_count_kept(self):
        model = {"size": 4_700_000_000,
                 "details": {"quantization_level": "Q4_K_M"}}
        info = {"general.parameter_count": 8_000_000_000}
        total, estimated, _ec, _eu = toks.ollama_param_count(model, info)
        self.assertEqual(total, 8_000_000_000)
        self.assertFalse(estimated)

    def test_bogus_count_replaced_by_size_estimate(self):
        # parameter_count implies ~72 bits/weight at this size -> discard it.
        model = {"size": 18_000_000_000,
                 "details": {"quantization_level": "Q4_K_M"}}
        info = {"general.parameter_count": 2_000_000_000}
        total, estimated, _ec, _eu = toks.ollama_param_count(model, info)
        self.assertTrue(estimated)
        self.assertNotEqual(total, 2_000_000_000)

    def test_parse_models_populates_param_fields(self):
        tags = [{"name": "m:8b", "size": 4_700_000_000,
                 "details": {"format": "gguf", "quantization_level": "Q4_K_M"}}]
        info = {"m:8b": {"general.parameter_count": 8_000_000_000}}
        rec = toks.ollama_parse_models(tags, info)[0]
        self.assertEqual(rec.param_count, 8_000_000_000)
        self.assertFalse(rec.param_estimated)
        self.assertEqual(rec.params, "8B")


# ---- GGUF header parsing ----------------------------------------------------


def _gguf_str(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _build_gguf(arch="gemma4", file_type=15, ctx=4096, tensors=None,
                expert_count=None, version=3):
    """Synthesize a minimal valid GGUF blob (header only; no tensor data)."""
    if tensors is None:
        tensors = [("token_embd.weight", [256, 64]), ("blk.0.weight", [64, 64])]
    kvs = [
        ("general.architecture", 8, _gguf_str(arch)),
        ("general.file_type", 5, struct.pack("<I", file_type)),
        (f"{arch}.context_length", 5, struct.pack("<I", ctx)),
    ]
    if expert_count is not None:
        kvs.append((f"{arch}.expert_count", 5, struct.pack("<I", expert_count)))
        kvs.append((f"{arch}.expert_used_count", 5, struct.pack("<I", 2)))
    blob = b"GGUF" + struct.pack("<I", version)
    blob += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
    for key, vtype, vbytes in kvs:
        blob += _gguf_str(key) + struct.pack("<I", vtype) + vbytes
    for offset, (name, dims) in enumerate(tensors):
        blob += _gguf_str(name) + struct.pack("<I", len(dims))
        for dim in dims:
            blob += struct.pack("<Q", dim)
        blob += struct.pack("<I", 0) + struct.pack("<Q", offset)  # type, offset
    return blob


class GgufMetaTests(unittest.TestCase):
    def _write(self, data):
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(data)
        tmp.close()
        self.addCleanup(lambda: pathlib.Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_exact_param_count_sums_tensor_dims(self):
        meta = toks.read_gguf_meta(self._write(
            _build_gguf(tensors=[("a", [256, 64]), ("b", [64, 64])])))
        self.assertEqual(meta["param_count"], 256 * 64 + 64 * 64)

    def test_quant_label_from_file_type(self):
        meta = toks.read_gguf_meta(self._write(_build_gguf(file_type=15)))
        self.assertEqual(meta["quant"], "Q4_K_M")

    def test_context_length_from_arch_key(self):
        meta = toks.read_gguf_meta(self._write(
            _build_gguf(arch="gemma4", ctx=262144)))
        self.assertEqual(meta["ctx"], 262144)

    def test_moe_expert_counts(self):
        meta = toks.read_gguf_meta(self._write(_build_gguf(expert_count=8)))
        self.assertEqual(meta["expert_count"], 8)
        self.assertEqual(meta["expert_used"], 2)

    def test_unknown_file_type_falls_back_to_label(self):
        meta = toks.read_gguf_meta(self._write(_build_gguf(file_type=999)))
        self.assertEqual(meta["quant"], "ftype999")

    def test_garbage_returns_none(self):
        self.assertIsNone(toks.read_gguf_meta(self._write(b"NOTGGUF-bytes")))

    def test_missing_file_returns_none(self):
        self.assertIsNone(toks.read_gguf_meta("/nonexistent/file.gguf"))


class SelectMainGgufTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.snap = pathlib.Path(tmp.name)

    def _f(self, name, size):
        (self.snap / name).write_bytes(b"x" * size)

    def test_picks_largest_non_projector(self):
        self._f("model-Q4_K_M.gguf", 5000)
        self._f("mmproj-F16.gguf", 9000)     # projector, ignored
        self._f("mtp-model.gguf", 8000)      # draft head, ignored
        self._f("model-Q2_K.gguf", 1000)
        self.assertEqual(toks.select_main_gguf(self.snap).name, "model-Q4_K_M.gguf")

    def test_none_when_only_projectors(self):
        self._f("mmproj-F16.gguf", 9000)
        self.assertIsNone(toks.select_main_gguf(self.snap))

    def test_none_when_empty(self):
        self.assertIsNone(toks.select_main_gguf(self.snap))


# ---- Unsloth Studio listing / enrich / benchmark / wiring -------------------


UNSLOTH_MODELS = {"object": "list",
                  "data": [{"id": "unsloth-org/demo-31B-it-GGUF",
                            "object": "model"}]}


def _make_gguf_repo(hub, repo_dir, gguf_bytes, extra=None):
    repo = hub / repo_dir
    (repo / "refs").mkdir(parents=True)
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")
    (snapshot / "model-Q4_K_M.gguf").write_bytes(gguf_bytes)
    for name, data in (extra or {}).items():
        (snapshot / name).write_bytes(data)
    return snapshot


class UnslothParseEnrichTests(unittest.TestCase):
    def test_parse_ids_only(self):
        recs = toks.unsloth_parse_models(UNSLOTH_MODELS)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].provider, "unsloth")
        self.assertTrue(recs[0].benchmarkable)

    def test_parse_empty_payload(self):
        self.assertEqual(toks.unsloth_parse_models({}), [])

    def test_enrich_from_cached_gguf(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        hub = pathlib.Path(tmp.name)
        gguf = _build_gguf(arch="gemma4", file_type=15, ctx=262144,
                           tensors=[("tok", [1000, 64]), ("out", [64, 1000])])
        _make_gguf_repo(hub, "models--unsloth-org--demo-31B-it-GGUF", gguf,
                        extra={"mmproj-F16.gguf": b"z" * 50})
        recs = toks.unsloth_parse_models(UNSLOTH_MODELS)
        toks.unsloth_enrich(recs, hub)
        rec = recs[0]
        self.assertEqual(rec.fmt, "gguf")
        self.assertEqual(rec.quant, "Q4_K_M")
        self.assertEqual(rec.ctx_max, 262144)
        self.assertEqual(rec.param_count, 1000 * 64 + 64 * 1000)
        self.assertFalse(rec.param_estimated)
        self.assertEqual(rec.size_bytes, len(gguf))   # main file, not mmproj
        self.assertIsNotNone(rec.modified_at)

    def test_enrich_unknown_repo_is_noop(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        recs = toks.unsloth_parse_models(UNSLOTH_MODELS)
        toks.unsloth_enrich(recs, pathlib.Path(tmp.name))   # must not raise
        self.assertIsNone(recs[0].size_bytes)


class LlamaCppTimingsTests(unittest.TestCase):
    def test_reads_predicted_per_second_and_ttft(self):
        result = toks.parse_llamacpp_timings(
            {"timings": {"predicted_per_second": 39.6, "prompt_ms": 540.0}})
        self.assertAlmostEqual(result.tokens_per_second, 39.6)
        self.assertAlmostEqual(result.time_to_first_token, 0.54)

    def test_missing_timings_returns_none(self):
        self.assertIsNone(toks.parse_llamacpp_timings({}))
        self.assertIsNone(toks.parse_llamacpp_timings({"timings": {}}))

    def test_zero_tps_returns_none(self):
        self.assertIsNone(toks.parse_llamacpp_timings(
            {"timings": {"predicted_per_second": 0}}))

    def test_no_prompt_ms_leaves_ttft_none(self):
        result = toks.parse_llamacpp_timings(
            {"timings": {"predicted_per_second": 10}})
        self.assertEqual(result.tokens_per_second, 10)
        self.assertIsNone(result.time_to_first_token)


class BenchUnslothSseTests(unittest.TestCase):
    CHUNKS = [
        {"choices": [{"index": 0, "text": "Hello"}]},
        {"choices": [{"index": 0, "text": " world"}]},
    ]

    def test_prefers_server_timings(self):
        final = {"choices": [{"index": 0, "text": "", "finish_reason": "length"}],
                 "usage": {"completion_tokens": 50},
                 "timings": {"predicted_per_second": 42.0, "prompt_ms": 200.0}}
        lines = _sse(*self.CHUNKS, final)
        result = toks.bench_unsloth_sse(lines, start=0.0,
                                        clock=FakeClock([1.0, 2.0]))
        self.assertAlmostEqual(result.tokens_per_second, 42.0)
        self.assertAlmostEqual(result.time_to_first_token, 0.2)

    def test_falls_back_to_client_timing(self):
        usage = {"choices": [], "usage": {"completion_tokens": 12}}
        lines = _sse(*self.CHUNKS, usage)
        result = toks.bench_unsloth_sse(lines, start=0.5,
                                        clock=FakeClock([1.0, 3.0]))
        self.assertAlmostEqual(result.tokens_per_second, 5.5)   # (12-1)/2.0
        self.assertAlmostEqual(result.time_to_first_token, 0.5)


class UnslothWiringTests(unittest.TestCase):
    def test_select_providers_unsloth(self):
        self.assertEqual([p.name for p in toks.select_providers("unsloth")],
                         ["unsloth"])

    def test_provider_flag_accepts_unsloth(self):
        self.assertEqual(toks.parse_args(["--provider", "unsloth"]).provider,
                         "unsloth")

    def test_default_url_and_cache_prefix(self):
        self.assertIn("unsloth:", toks._PROVIDER_PREFIXES)
        with mock.patch.dict(toks.os.environ, {}, clear=True):
            self.assertEqual(toks.unsloth_url(), "http://127.0.0.1:8888")

    def test_headers_from_api_key(self):
        with mock.patch.dict(toks.os.environ, {"UNSLOTH_API_KEY": "sk-zzz"},
                             clear=True):
            self.assertEqual(toks.unsloth_headers(),
                             {"Authorization": "Bearer sk-zzz"})
        with mock.patch.dict(toks.os.environ, {}, clear=True):
            self.assertEqual(toks.unsloth_headers(), {})


# ---- .env loader (read-only) -----------------------------------------------


class LoadDotenvTests(unittest.TestCase):
    def _write_env(self, text):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        tmp.write(text)
        tmp.close()
        self.addCleanup(lambda: pathlib.Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_sets_keys_stripping_quotes_comments_export(self):
        path = self._write_env(
            "# a comment\n"
            "\n"
            "UNSLOTH_API_KEY=sk-test-123\n"
            'export UNSLOTH_URL="http://127.0.0.1:8888"\n'
            "QUOTED='single'\n")
        with mock.patch.dict(toks.os.environ, {}, clear=True):
            toks.load_dotenv(path)
            self.assertEqual(toks.os.environ["UNSLOTH_API_KEY"], "sk-test-123")
            self.assertEqual(toks.os.environ["UNSLOTH_URL"], "http://127.0.0.1:8888")
            self.assertEqual(toks.os.environ["QUOTED"], "single")

    def test_real_env_var_wins(self):
        path = self._write_env("UNSLOTH_API_KEY=from-file\n")
        with mock.patch.dict(toks.os.environ, {"UNSLOTH_API_KEY": "from-env"},
                             clear=True):
            toks.load_dotenv(path)
            self.assertEqual(toks.os.environ["UNSLOTH_API_KEY"], "from-env")

    def test_missing_file_is_noop(self):
        with mock.patch.dict(toks.os.environ, {}, clear=True):
            toks.load_dotenv("/nonexistent/dir/.env")   # must not raise
            self.assertNotIn("UNSLOTH_API_KEY", toks.os.environ)


if __name__ == "__main__":
    unittest.main()
