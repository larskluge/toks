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

    def test_degenerate_one_token_generation_rejected(self):
        # 1 token then EOS: lmstudio reports tps over ~0s -> a nonsense number
        # (e.g. 142857 tok/s). A non-representative run must not be cached.
        self.assertIsNone(toks.parse_lmstudio_bench(
            {"stats": {"tokens_per_second": 142857.0, "generation_time": 0},
             "usage": {"completion_tokens": 1}}))

    def test_full_generation_with_usage_kept(self):
        result = toks.parse_lmstudio_bench(
            {"stats": {"tokens_per_second": 51.0}, "usage": {"completion_tokens": 200}})
        self.assertAlmostEqual(result.tokens_per_second, 51.0)


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

    def test_degenerate_one_token_generation_rejected(self):
        # 1 token in ~1us is a degenerate run, not a throughput measurement.
        self.assertIsNone(toks.parse_ollama_bench(
            {"eval_count": 1, "eval_duration": 1000}))


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
                         ["ollama", "lmstudio", "mlx", "unsloth", "llama"])

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
        lines = out.splitlines()
        self.assertIn("PROVIDER", lines[0])
        self.assertIn(" │ ", lines[0])               # column separators
        self.assertEqual(set(lines[1]), {"─", "┼"})   # header/body rule
        self.assertEqual(len(lines), 4)               # header + rule + 2 rows


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

    def test_samples_flag_default_and_override(self):
        self.assertEqual(toks.parse_args([]).samples, toks.DEFAULT_SAMPLES)
        self.assertEqual(toks.parse_args(["--samples", "5"]).samples, 5)

    def test_samples_rejects_non_positive(self):
        self._error(["--samples", "0"])
        self._error(["--samples", "-3"])


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


class NoncePromptTests(unittest.TestCase):
    def test_two_calls_differ(self):
        self.assertNotEqual(toks.nonce_prompt("hi"), toks.nonce_prompt("hi"))

    def test_base_prompt_preserved_as_suffix(self):
        self.assertTrue(toks.nonce_prompt("the base").endswith("the base"))

    def test_unique_bytes_lead(self):
        a, b = toks.nonce_prompt("BASE"), toks.nonce_prompt("BASE")
        lead_a, lead_b = a[: -len("BASE")], b[: -len("BASE")]
        self.assertNotEqual(lead_a, lead_b)        # the differing part leads
        self.assertTrue(lead_a.startswith("["))    # nonce is a bracketed prefix


class AggregateSamplesTests(unittest.TestCase):
    def test_median_min_max_odd(self):
        agg = toks.aggregate_samples(
            [toks.BenchResult(x, 0.1) for x in (40.0, 60.0, 50.0)])
        self.assertAlmostEqual(agg["tokens_per_second"], 50.0)
        self.assertAlmostEqual(agg["tps_min"], 40.0)
        self.assertAlmostEqual(agg["tps_max"], 60.0)
        self.assertEqual(agg["samples"], 3)

    def test_median_even(self):
        agg = toks.aggregate_samples([toks.BenchResult(x) for x in (40.0, 50.0)])
        self.assertAlmostEqual(agg["tokens_per_second"], 45.0)

    def test_single_sample(self):
        agg = toks.aggregate_samples([toks.BenchResult(42.0, 0.1)])
        self.assertAlmostEqual(agg["tokens_per_second"], 42.0)
        self.assertAlmostEqual(agg["tps_min"], 42.0)
        self.assertAlmostEqual(agg["tps_max"], 42.0)
        self.assertEqual(agg["samples"], 1)

    def test_ttft_median_ignores_missing(self):
        results = [toks.BenchResult(40.0, 0.1), toks.BenchResult(50.0, None),
                   toks.BenchResult(60.0, 0.3)]
        self.assertAlmostEqual(
            toks.aggregate_samples(results)["time_to_first_token"], 0.2)

    def test_ttft_none_when_all_missing(self):
        agg = toks.aggregate_samples([toks.BenchResult(40.0), toks.BenchResult(50.0)])
        self.assertIsNone(agg["time_to_first_token"])

    def test_ttft_zero_is_a_reported_value_not_missing(self):
        # A reported 0.0 is data, not absence (absence is None) -- keep it.
        results = [toks.BenchResult(40.0, 0.0), toks.BenchResult(60.0, 0.2)]
        self.assertAlmostEqual(
            toks.aggregate_samples(results)["time_to_first_token"], 0.1)

    def test_pooled_acceptance(self):
        results = [toks.BenchResult(40.0, draft_n=48, draft_n_accepted=46),
                   toks.BenchResult(50.0, draft_n=52, draft_n_accepted=40)]
        agg = toks.aggregate_samples(results)
        self.assertEqual(agg["draft_n"], 100)            # pooled drafted
        self.assertEqual(agg["draft_n_accepted"], 86)    # pooled accepted

    def test_pooled_acceptance_mixed_with_and_without_draft(self):
        # One spec-decode sample, one plain sample: pooled counters ignore the
        # plain one (None treated as 0) rather than corrupting the ratio.
        results = [toks.BenchResult(40.0, draft_n=48, draft_n_accepted=46),
                   toks.BenchResult(50.0)]
        agg = toks.aggregate_samples(results)
        self.assertEqual(agg["draft_n"], 48)
        self.assertEqual(agg["draft_n_accepted"], 46)

    def test_draft_none_when_no_sample_carries_it(self):
        agg = toks.aggregate_samples([toks.BenchResult(40.0), toks.BenchResult(50.0)])
        self.assertIsNone(agg["draft_n"])
        self.assertIsNone(agg["draft_n_accepted"])


class BenchDefaultsTests(unittest.TestCase):
    def test_default_max_tokens_is_256(self):
        self.assertEqual(toks.DEFAULT_MAX_TOKENS, 256)

    def test_default_samples_is_3(self):
        self.assertEqual(toks.DEFAULT_SAMPLES, 3)


class RunBenchmarksTests(unittest.TestCase):
    class FakeProvider:
        def __init__(self, results=None):
            self.calls = []
            self._results = list(results) if results is not None else None

        def benchmark(self, rec, prompt, max_tokens):
            self.calls.append((rec.name, prompt, max_tokens))
            if self._results is not None:
                return self._results.pop(0)
            return toks.BenchResult(42.0, 0.1)

    def test_runs_n_samples_with_distinct_nonced_prompts(self):
        rec = toks.ModelRecord("ollama", "scribe:8b", digest="aaaa1111",
                               benchmarkable=True)
        provider = self.FakeProvider([toks.BenchResult(40.0, 0.1),
                                      toks.BenchResult(50.0, 0.2),
                                      toks.BenchResult(60.0, 0.3)])
        cache = {}
        with contextlib.redirect_stderr(io.StringIO()):
            changed = toks.run_benchmarks([rec], {"ollama": provider}, cache)
        self.assertTrue(changed)
        self.assertEqual(len(provider.calls), 3)               # DEFAULT_SAMPLES
        prompts = [p for (_, p, _) in provider.calls]
        self.assertEqual(len(set(prompts)), 3)                 # each nonce distinct
        self.assertTrue(all(p.endswith(toks.DEFAULT_PROMPT) for p in prompts))
        self.assertEqual({mt for (_, _, mt) in provider.calls},
                         {toks.DEFAULT_MAX_TOKENS})
        entry = cache["ollama:aaaa1111"]
        self.assertAlmostEqual(entry["tokens_per_second"], 50.0)   # median
        self.assertEqual(entry["samples"], 3)
        self.assertAlmostEqual(entry["tps_min"], 40.0)
        self.assertAlmostEqual(entry["tps_max"], 60.0)
        self.assertEqual(entry["prompt"], toks.DEFAULT_PROMPT)     # nonce stripped
        self.assertTrue(entry["prompt_nonced"])
        self.assertEqual(entry["max_tokens"], toks.DEFAULT_MAX_TOKENS)

    def test_samples_one_reproduces_single_shot(self):
        rec = toks.ModelRecord("ollama", "scribe:8b", digest="aaaa1111",
                               benchmarkable=True)
        provider = self.FakeProvider([toks.BenchResult(42.0, 0.1)])
        cache = {}
        with contextlib.redirect_stderr(io.StringIO()):
            toks.run_benchmarks([rec], {"ollama": provider}, cache, samples=1)
        self.assertEqual(len(provider.calls), 1)
        entry = cache["ollama:aaaa1111"]
        self.assertAlmostEqual(entry["tokens_per_second"], 42.0)
        self.assertEqual(entry["samples"], 1)
        self.assertAlmostEqual(entry["tps_min"], 42.0)
        self.assertAlmostEqual(entry["tps_max"], 42.0)

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
        statuses = [("ollama", True), ("lmstudio", True)]
        out = io.StringIO()
        with mock.patch.object(toks, "gather_records",
                               return_value=(records, active, statuses)), \
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


# ---- provider reachability summary -----------------------------------------


class _StubProvider:
    def __init__(self, name, records=None, fail=False):
        self.name = name
        self._records = records or []
        self._fail = fail

    def list_models(self):
        if self._fail:
            raise RuntimeError("could not reach http://localhost:8080/v1/models")
        return self._records


class GatherRecordsTests(unittest.TestCase):
    def test_reachable_and_failed_providers_reported_in_order(self):
        rec = toks.ModelRecord("ollama", "m")
        providers = [
            _StubProvider("ollama", records=[rec]),
            _StubProvider("mlx", fail=True),
        ]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            records, active, statuses = toks.gather_records(providers)
        self.assertEqual(records, [rec])
        self.assertEqual(list(active), ["ollama"])
        self.assertEqual(statuses, [("ollama", True), ("mlx", False)])

    def test_failure_does_not_print_the_long_warning(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            toks.gather_records([_StubProvider("mlx", fail=True)])
        self.assertEqual(err.getvalue(), "")


class ProviderStatusLineTests(unittest.TestCase):
    def test_plain_line_marks_reachable_and_unreachable(self):
        line = toks.provider_status_line(
            [("ollama", True), ("mlx", False)], color=False)
        self.assertEqual(line, "ollama ✓  mlx ✗")
        self.assertNotIn("\x1b", line)

    def test_single_line_for_all_providers(self):
        statuses = [("ollama", True), ("lmstudio", False),
                    ("mlx", False), ("unsloth", True)]
        line = toks.provider_status_line(statuses, color=False)
        self.assertNotIn("\n", line)

    def test_color_wraps_marks_only(self):
        line = toks.provider_status_line(
            [("ollama", True), ("mlx", False)], color=True)
        self.assertIn(f"\x1b[{toks.STATUS_OK}m✓\x1b[0m", line)
        self.assertIn(f"\x1b[{toks.STATUS_BAD}m✗\x1b[0m", line)
        self.assertTrue(line.startswith("ollama "))   # names stay uncoloured


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


# Anonymized capture of GET /api/hub/local: HF-cache, LM Studio, and Ollama
# entries carrying Studio's (deliberately imperfect) capability flags.
HUB_LOCAL = {"models": [
    {"id": "acme/demo-31B-it-GGUF", "source": "hf_cache",
     "model_format": "gguf", "runtime": "llama_cpp",
     "size_bytes": 18_000_000_000, "capabilities": {"can_chat": True}},
    {"id": "mlx-folk/demo-coder-8bit", "source": "hf_cache",
     "model_format": "safetensors", "runtime": "transformers",
     "size_bytes": 8_000_000_000, "capabilities": {"can_chat": True}},
    {"id": "acme/bge-small-en-GGUF", "source": "hf_cache",          # embedding...
     "model_format": "gguf", "runtime": "llama_cpp",
     "capabilities": {"can_chat": True}},                          # ...mislabeled
    {"id": "voicelab/tts-1.6b", "source": "hf_cache",              # speech...
     "model_format": "safetensors", "runtime": "transformers",
     "capabilities": {"can_chat": True}},                          # ...mislabeled
    {"id": "imagelab/Z-Image-Turbo", "source": "hf_cache",         # image gen
     "model_format": "safetensors", "runtime": "transformers",
     "capabilities": {"can_chat": False}},
    {"id": "ggml-folk/models", "source": "hf_cache",               # partial junk
     "model_format": "unknown", "runtime": "unknown",
     "capabilities": {"can_chat": False}},
    {"id": "/home/u/.lmstudio/models/acme/demo-35B", "source": "lmstudio",
     "model_format": "gguf", "runtime": "llama_cpp",
     "capabilities": {"can_chat": True}},                          # LM Studio's own
    {"id": "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmodels%2Fdemo",
     "source": "ollama", "model_format": "gguf", "runtime": "llama_cpp",
     "capabilities": {"can_chat": True}},                          # Ollama's own
]}

# Single hf_cache entry whose id maps to the cached repo dir built in tests.
ENRICH_LOCAL = {"models": [{"id": "acme/demo-31B-it-GGUF", "source": "hf_cache",
                            "model_format": "gguf", "runtime": "llama_cpp",
                            "capabilities": {"can_chat": True}}]}


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


class UnslothParseLocalTests(unittest.TestCase):
    def test_keeps_hf_cache_text_models(self):
        recs = toks.unsloth_parse_local(HUB_LOCAL)
        self.assertEqual({r.name for r in recs},
                         {"acme/demo-31B-it-GGUF", "mlx-folk/demo-coder-8bit"})
        self.assertTrue(all(r.provider == "unsloth" and r.benchmarkable
                            for r in recs))

    def test_drops_other_sources(self):
        names = {r.name for r in toks.unsloth_parse_local(HUB_LOCAL)}
        self.assertNotIn("/home/u/.lmstudio/models/acme/demo-35B", names)
        self.assertFalse(any("ollama-manifest" in n for n in names))

    def test_drops_nonchat_and_unknown_format(self):
        names = {r.name for r in toks.unsloth_parse_local(HUB_LOCAL)}
        self.assertNotIn("imagelab/Z-Image-Turbo", names)   # can_chat False
        self.assertNotIn("ggml-folk/models", names)         # unknown format

    def test_drops_embedding_and_speech_despite_can_chat(self):
        names = {r.name for r in toks.unsloth_parse_local(HUB_LOCAL)}
        self.assertNotIn("acme/bge-small-en-GGUF", names)
        self.assertNotIn("voicelab/tts-1.6b", names)

    def test_keeps_non_unsloth_namespace_overlap(self):
        # An mlx-community-style repo is listed under unsloth on purpose, so its
        # Studio throughput can be compared against the mlx provider's number.
        names = {r.name for r in toks.unsloth_parse_local(HUB_LOCAL)}
        self.assertIn("mlx-folk/demo-coder-8bit", names)

    def test_empty_or_malformed(self):
        self.assertEqual(toks.unsloth_parse_local({}), [])
        self.assertEqual(toks.unsloth_parse_local({"models": "nope"}), [])

    def test_enrich_from_cached_gguf(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        hub = pathlib.Path(tmp.name)
        gguf = _build_gguf(arch="gemma4", file_type=15, ctx=262144,
                           tensors=[("tok", [1000, 64]), ("out", [64, 1000])])
        _make_gguf_repo(hub, "models--acme--demo-31B-it-GGUF", gguf,
                        extra={"mmproj-F16.gguf": b"z" * 50})
        recs = toks.unsloth_parse_local(ENRICH_LOCAL)
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
        recs = toks.unsloth_parse_local(ENRICH_LOCAL)
        toks.unsloth_enrich(recs, pathlib.Path(tmp.name))   # must not raise
        self.assertIsNone(recs[0].size_bytes)

    def test_enrich_mlx_safetensors_fallback(self):
        # An mlx-community-style repo (no GGUF) enriches from config.json so its
        # cross-runtime-comparison row isn't all dashes.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        hub = pathlib.Path(tmp.name)
        config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                  "max_position_embeddings": 262144,
                  "quantization": {"group_size": 64, "bits": 8}}
        _make_hub_repo(hub, "models--mlx-folk--demo-coder-8bit", config)
        local = {"models": [{"id": "mlx-folk/demo-coder-8bit", "source": "hf_cache",
                             "model_format": "safetensors", "runtime": "transformers",
                             "capabilities": {"can_chat": True}}]}
        recs = toks.unsloth_parse_local(local)
        toks.unsloth_enrich(recs, hub)
        rec = recs[0]
        self.assertEqual(rec.fmt, "safetensors")
        self.assertEqual(rec.quant, "8bit")
        self.assertEqual(rec.ctx_max, 262144)
        self.assertTrue(rec.param_estimated)          # size-derived, not exact
        self.assertTrue(rec.params.startswith("~"))
        self.assertIsNotNone(rec.size_bytes)
        self.assertIsNotNone(rec.modified_at)


class UnslothListModelsTests(unittest.TestCase):
    def test_queries_hub_local(self):
        provider = toks.UnslothProvider()
        with mock.patch.object(toks, "http_json", return_value=ENRICH_LOCAL) as hj, \
                mock.patch.object(toks, "is_local_host", return_value=False):
            recs = provider.list_models()
        self.assertEqual([r.name for r in recs], ["acme/demo-31B-it-GGUF"])
        self.assertTrue(hj.call_args[0][0].endswith("/api/hub/local"))


class ServedModelMatchesTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(toks.served_model_matches("acme/demo-31B-it-GGUF",
                                                   "acme/demo-31B-it-GGUF"))

    def test_case_insensitive(self):
        self.assertTrue(toks.served_model_matches("Acme/Demo", "acme/demo"))

    def test_basename_match_ignores_org_or_path_prefix(self):
        # Studio may report the loaded model's bare alias without the org prefix.
        self.assertTrue(toks.served_model_matches("acme/demo-31B-it-GGUF",
                                                   "demo-31B-it-GGUF"))

    def test_different_model_is_mismatch(self):
        self.assertFalse(toks.served_model_matches("acme/demo-31B-it-GGUF",
                                                    "other/llama-3b"))

    def test_hf_cache_path_served_matches_repo_id(self):
        # Studio reports the loaded model as its on-disk HF-cache path; the
        # `models--org--name` segment encodes the requested repo id, and the
        # filename carries an extra quant suffix that bare basename matching misses.
        served = ("/home/u/.cache/huggingface/hub/models--acme--demo-31B-it-GGUF/"
                  "snapshots/abc123/demo-31B-it-UD-Q4_K_XL.gguf")
        self.assertTrue(toks.served_model_matches("acme/demo-31B-it-GGUF", served))

    def test_hf_cache_path_for_other_model_is_mismatch(self):
        served = ("/home/u/.cache/huggingface/hub/models--other--llama-3b/"
                  "snapshots/abc123/llama-3b-Q4_K_M.gguf")
        self.assertFalse(toks.served_model_matches("acme/demo-31B-it-GGUF", served))

    def test_unknown_served_cannot_be_verified(self):
        # No served id (server didn't echo one) -> don't block the benchmark.
        self.assertTrue(toks.served_model_matches("acme/demo", ""))
        self.assertTrue(toks.served_model_matches("acme/demo", None))


class UnslothBenchGuardTests(unittest.TestCase):
    """Studio serves its one loaded model regardless of the requested name, so a
    benchmark must verify the served model before trusting (and caching) timings.
    """

    def _rec(self):
        return toks.ModelRecord("unsloth", "acme/demo-31B-it-GGUF")

    def test_skips_when_server_serves_a_different_model(self):
        provider = toks.UnslothProvider()
        warm = {"model": "other/small-model"}
        with mock.patch.object(toks, "http_json", return_value=warm), \
                mock.patch.object(toks, "http_sse") as sse:
            with self.assertRaises(RuntimeError) as ctx:
                provider.benchmark(self._rec(), "p", 8)
        self.assertIn("other/small-model", str(ctx.exception))
        sse.assert_not_called()              # never ran the timed stream

    def test_proceeds_when_served_model_matches(self):
        provider = toks.UnslothProvider()
        warm = {"model": "acme/demo-31B-it-GGUF"}
        stream = _sse({"choices": [{"text": "a"}]},
                      {"choices": [], "usage": {"completion_tokens": 50},
                       "timings": {"predicted_per_second": 42.0, "prompt_ms": 100.0}})
        with mock.patch.object(toks, "http_json", return_value=warm), \
                mock.patch.object(toks, "http_sse", return_value=stream):
            result = provider.benchmark(self._rec(), "p", 8)
        self.assertAlmostEqual(result.tokens_per_second, 42.0)

    def test_proceeds_when_served_model_unknown(self):
        # A server that doesn't echo a model id can't be verified -> don't block.
        provider = toks.UnslothProvider()
        stream = _sse({"choices": [{"text": "a"}]},
                      {"choices": [], "usage": {"completion_tokens": 50},
                       "timings": {"predicted_per_second": 7.0}})
        with mock.patch.object(toks, "http_json", return_value={"no": "model"}), \
                mock.patch.object(toks, "http_sse", return_value=stream):
            result = provider.benchmark(self._rec(), "p", 8)
        self.assertAlmostEqual(result.tokens_per_second, 7.0)

    def test_proceeds_when_warmup_fails(self):
        # Warmup error (can't read served model) must not block benchmarking.
        provider = toks.UnslothProvider()
        stream = _sse({"choices": [{"text": "a"}]},
                      {"choices": [], "usage": {"completion_tokens": 50},
                       "timings": {"predicted_per_second": 5.0}})
        with mock.patch.object(toks, "http_json",
                               side_effect=RuntimeError("warmup boom")), \
                mock.patch.object(toks, "http_sse", return_value=stream):
            result = provider.benchmark(self._rec(), "p", 8)
        self.assertAlmostEqual(result.tokens_per_second, 5.0)

    def test_falls_back_to_chat_completions_for_mlx(self):
        # /v1/completions 503s for MLX models; fall back to /v1/chat/completions.
        provider = toks.UnslothProvider()
        # completions warmup raises (GGUF-only 503); chat warmup echoes the model.
        warm = mock.Mock(side_effect=[RuntimeError("HTTP 503"),
                                      {"model": "acme/demo-31B-it-GGUF"}])
        chat_stream = _sse({"choices": [{"delta": {"content": "a"}}]},
                           {"choices": [{"delta": {"content": "b"}}]},
                           {"choices": [], "usage": {"completion_tokens": 20}})
        with mock.patch.object(toks, "http_json", warm), \
                mock.patch.object(toks, "http_sse",
                                  return_value=chat_stream) as sse:
            result = provider.benchmark(self._rec(), "p", 8)
        self.assertIsNotNone(result)
        # streamed against the chat endpoint with a messages payload, not a prompt.
        url, payload = sse.call_args[0][0], sse.call_args[0][1]
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertIn("messages", payload)
        self.assertNotIn("prompt", payload)

    def test_chat_fallback_still_guards_mismatch(self):
        # Even on the chat path, a mismatched served model is skipped.
        provider = toks.UnslothProvider()
        warm = mock.Mock(side_effect=[RuntimeError("HTTP 503"),
                                      {"model": "other/small-model"}])
        with mock.patch.object(toks, "http_json", warm), \
                mock.patch.object(toks, "http_sse") as sse:
            with self.assertRaises(RuntimeError) as ctx:
                provider.benchmark(self._rec(), "p", 8)
        self.assertIn("other/small-model", str(ctx.exception))
        sse.assert_not_called()


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

    def test_lifts_draft_and_cache_counters(self):
        # The real Studio /v1/completions timings block (verified 2026-06-27).
        result = toks.parse_llamacpp_timings({"timings": {
            "cache_n": 1, "prompt_n": 4, "prompt_ms": 19.8, "predicted_n": 64,
            "predicted_ms": 366.7, "predicted_per_second": 174.5,
            "draft_n": 48, "draft_n_accepted": 46}})
        self.assertEqual(result.draft_n, 48)
        self.assertEqual(result.draft_n_accepted, 46)
        self.assertEqual(result.cache_n, 1)

    def test_draft_and_cache_none_when_absent(self):
        result = toks.parse_llamacpp_timings(
            {"timings": {"predicted_per_second": 10}})
        self.assertIsNone(result.draft_n)
        self.assertIsNone(result.draft_n_accepted)
        self.assertIsNone(result.cache_n)


class BenchLlamacppSseTests(unittest.TestCase):
    CHUNKS = [
        {"choices": [{"index": 0, "text": "Hello"}]},
        {"choices": [{"index": 0, "text": " world"}]},
    ]

    def test_prefers_server_timings(self):
        final = {"choices": [{"index": 0, "text": "", "finish_reason": "length"}],
                 "usage": {"completion_tokens": 50},
                 "timings": {"predicted_per_second": 42.0, "prompt_ms": 200.0}}
        lines = _sse(*self.CHUNKS, final)
        result = toks.bench_llamacpp_sse(lines, start=0.0,
                                         clock=FakeClock([1.0, 2.0]))
        self.assertAlmostEqual(result.tokens_per_second, 42.0)
        self.assertAlmostEqual(result.time_to_first_token, 0.2)

    def test_falls_back_to_client_timing(self):
        usage = {"choices": [], "usage": {"completion_tokens": 12}}
        lines = _sse(*self.CHUNKS, usage)
        result = toks.bench_llamacpp_sse(lines, start=0.5,
                                         clock=FakeClock([1.0, 3.0]))
        self.assertAlmostEqual(result.tokens_per_second, 5.5)   # (12-1)/2.0
        self.assertAlmostEqual(result.time_to_first_token, 0.5)

    def test_otherwise_defaults_to_unknown_for_unsloth(self):
        # No timings, no fingerprint: the unsloth caller's default labels it unknown.
        lines = _sse(*self.CHUNKS, {"choices": [], "usage": {"completion_tokens": 6}})
        r = toks.bench_llamacpp_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]))
        self.assertEqual(r.observed_backend, "unknown")

    def test_otherwise_llamacpp_for_plain_llama_server(self):
        # A plain llama-server passes otherwise="llamacpp" for the (unreachable in
        # practice) no-timings fallback.
        lines = _sse(*self.CHUNKS, {"choices": [], "usage": {"completion_tokens": 6}})
        r = toks.bench_llamacpp_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]),
                                    otherwise="llamacpp")
        self.assertEqual(r.observed_backend, "llamacpp")


class ObservedBackendTests(unittest.TestCase):
    """Each bench path records which engine actually answered."""

    def test_fingerprint_classifier(self):
        self.assertTrue(toks._looks_llamacpp_fingerprint("b9773-abc123"))
        self.assertFalse(toks._looks_llamacpp_fingerprint("fp_44709c"))
        self.assertFalse(toks._looks_llamacpp_fingerprint(None))

    def test_ollama_parser_stamps_backend(self):
        r = toks.parse_ollama_bench({"eval_count": 10, "eval_duration": 1_000_000_000})
        self.assertEqual((r.tps_source, r.observed_backend), ("ollama", "ollama"))

    def test_lmstudio_parser_stamps_backend_and_fingerprint(self):
        r = toks.parse_lmstudio_bench({"stats": {"tokens_per_second": 20.0},
                                       "system_fingerprint": "lmstudio-fp"})
        self.assertEqual((r.tps_source, r.observed_backend),
                         ("lmstudio_stats", "lmstudio"))
        self.assertEqual(r.system_fingerprint, "lmstudio-fp")

    def test_llamacpp_timings_stamps_backend_and_fingerprint(self):
        r = toks.parse_llamacpp_timings({"timings": {"predicted_per_second": 30.0},
                                         "system_fingerprint": "b9773-deadbeef"})
        self.assertEqual((r.tps_source, r.observed_backend),
                         ("llamacpp_timings", "llamacpp"))
        self.assertEqual(r.system_fingerprint, "b9773-deadbeef")

    def test_mlx_stream_is_mlx_lm(self):
        lines = _sse({"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]},
                     {"choices": [], "usage": {"completion_tokens": 6}})
        r = toks.bench_from_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]))
        self.assertEqual(r.observed_backend, "mlx_lm")
        self.assertEqual(r.tps_source, "client_timed")

    def test_mlx_port_serving_llamacpp_flagged_by_timings(self):
        # A llama.cpp server squatting on the mlx port leaks a timings block.
        lines = _sse({"choices": [{"text": "a"}]},
                     {"choices": [{"text": "b"}],
                      "timings": {"predicted_per_second": 9}},
                     {"choices": [], "usage": {"completion_tokens": 6}})
        r = toks.bench_from_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]))
        self.assertEqual(r.observed_backend, "llamacpp")

    def test_mlx_port_serving_llamacpp_flagged_by_fingerprint(self):
        lines = _sse({"choices": [{"text": "a"}], "system_fingerprint": "b9773-x"},
                     {"choices": [{"text": "b"}]},
                     {"choices": [], "usage": {"completion_tokens": 6}})
        r = toks.bench_from_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]))
        self.assertEqual(r.observed_backend, "llamacpp")

    def test_unsloth_fallback_backend_is_unknown(self):
        # No llama.cpp timings: Studio served via another runtime (transformers).
        lines = _sse({"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]},
                     {"choices": [], "usage": {"completion_tokens": 6}})
        r = toks.bench_llamacpp_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]))
        self.assertEqual((r.tps_source, r.observed_backend),
                         ("client_timed", "unknown"))


class BenchIdentityPersistenceTests(unittest.TestCase):
    class StubProvider:
        def __init__(self, host, expected, result):
            self.host = host
            self.expected_backend = expected
            self._result = result

        def benchmark(self, rec, prompt, max_tokens):
            return self._result

    def _bench(self, provider_name, provider):
        rec = toks.ModelRecord(provider_name, "demo-model", benchmarkable=True)
        cache = {}
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            toks.run_benchmarks([rec], {provider_name: provider}, cache)
        return cache[toks.cache_key(rec)], err.getvalue()

    def test_records_observed_backend_and_endpoint(self):
        result = toks.BenchResult(50.0, 0.2, tps_source="client_timed",
                                  observed_backend="mlx_lm")
        entry, err = self._bench(
            "mlx", self.StubProvider("http://localhost:8080", "mlx_lm", result))
        self.assertEqual(entry["endpoint"], "http://localhost:8080")
        self.assertEqual(entry["observed_backend"], "mlx_lm")
        self.assertEqual(entry["tps_source"], "client_timed")
        self.assertNotIn("warning", err)

    def test_warns_when_observed_differs_from_expected(self):
        result = toks.BenchResult(50.0, 0.2, tps_source="client_timed",
                                  observed_backend="llamacpp",
                                  system_fingerprint="b9773-x")
        entry, err = self._bench(
            "mlx", self.StubProvider("http://localhost:8080", "mlx_lm", result))
        self.assertEqual(entry["observed_backend"], "llamacpp")
        self.assertIn("expected mlx_lm", err)
        self.assertIn("observed llamacpp", err)
        self.assertIn("b9773-x", err)

    def test_no_warning_when_provider_expects_none(self):
        result = toks.BenchResult(50.0, 0.2, tps_source="client_timed",
                                  observed_backend="unknown")
        entry, err = self._bench(
            "unsloth", self.StubProvider("http://127.0.0.1:8888", None, result))
        self.assertEqual(entry["observed_backend"], "unknown")
        self.assertNotIn("warning", err)


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


# ---- llama.cpp server listing / enrich / wiring ----------------------------

# Anonymized capture of llama-server GET /v1/models: we read the OpenAI-style
# `data` array, whose `meta` block carries exact params, on-disk size, and ctx.
LLAMA_MODELS = {"object": "list", "data": [
    {"id": "acme/demo-31B-it-GGUF", "owned_by": "llamacpp",
     "meta": {"n_ctx": 131072, "n_ctx_train": 262144,
              "n_params": 30_000_000_000, "size": 17_000_000_000}},
]}


class LlamaListingTests(unittest.TestCase):
    def test_parses_data_into_records(self):
        rec = toks.llama_parse_models(LLAMA_MODELS)[0]
        self.assertEqual((rec.provider, rec.name, rec.fmt),
                         ("llama", "acme/demo-31B-it-GGUF", "gguf"))
        self.assertTrue(rec.benchmarkable)

    def test_meta_populates_params_size_and_ctx(self):
        rec = toks.llama_parse_models(LLAMA_MODELS)[0]
        self.assertEqual(rec.param_count, 30_000_000_000)
        self.assertFalse(rec.param_estimated)             # server-reported, exact
        self.assertEqual(rec.params, toks.human_params(30_000_000_000, None, None))
        self.assertFalse(rec.params.startswith("~"))
        self.assertEqual(rec.size_bytes, 17_000_000_000)
        self.assertEqual(rec.ctx_loaded, 131072)          # running ctx
        self.assertEqual(rec.ctx_max, 262144)             # trained max

    def test_quant_absent_until_enriched(self):
        rec = toks.llama_parse_models(LLAMA_MODELS)[0]
        self.assertEqual(rec.quant, "-")                  # only the GGUF has it

    def test_missing_or_malformed_meta_leaves_defaults(self):
        rec = toks.llama_parse_models({"data": [{"id": "x"}]})[0]
        self.assertEqual((rec.name, rec.fmt), ("x", "gguf"))
        self.assertIsNone(rec.param_count)
        self.assertIsNone(rec.size_bytes)
        self.assertIsNone(rec.ctx_loaded)

    def test_empty_or_missing_data_yields_no_records(self):
        self.assertEqual(toks.llama_parse_models({}), [])
        self.assertEqual(toks.llama_parse_models({"data": []}), [])
        self.assertEqual(toks.llama_parse_models({"data": "nope"}), [])
        self.assertEqual(toks.llama_parse_models({"data": [{"no": "id"}]}), [])


class LlamaEnrichmentTests(unittest.TestCase):
    def _gguf_file(self, **kw):
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(_build_gguf(**kw))
        tmp.close()
        self.addCleanup(lambda: pathlib.Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_backfills_quant_params_ctx_and_modified(self):
        path = self._gguf_file(arch="gemma4", file_type=15, ctx=262144,
                               tensors=[("tok", [1000, 64]), ("out", [64, 1000])])
        recs = toks.llama_parse_models(LLAMA_MODELS)
        props = {"model_alias": "acme/demo-31B-it-GGUF", "model_path": path}
        toks.llama_enrich(recs, props)
        rec = recs[0]
        self.assertEqual(rec.quant, "Q4_K_M")             # only the GGUF supplies it
        self.assertEqual(rec.ctx_max, 262144)
        self.assertEqual(rec.param_count, 1000 * 64 + 64 * 1000)  # GGUF refines meta
        self.assertFalse(rec.param_estimated)
        self.assertFalse(rec.moe)
        self.assertEqual(rec.size_bytes, pathlib.Path(path).stat().st_size)
        self.assertIsNotNone(rec.modified_at)

    def test_moe_detected_from_gguf(self):
        path = self._gguf_file(expert_count=8)
        recs = toks.llama_parse_models(LLAMA_MODELS)
        toks.llama_enrich(recs, {"model_alias": "acme/demo-31B-it-GGUF",
                                 "model_path": path})
        self.assertTrue(recs[0].moe)

    def test_alias_selects_the_matching_record(self):
        path = self._gguf_file()
        recs = [toks.ModelRecord("llama", "other/model"),
                toks.ModelRecord("llama", "acme/demo-31B-it-GGUF")]
        toks.llama_enrich(recs, {"model_alias": "acme/demo-31B-it-GGUF",
                                 "model_path": path})
        self.assertEqual(recs[0].quant, "-")              # untouched
        self.assertEqual(recs[1].fmt, "gguf")             # the loaded one

    def test_single_record_fallback_without_alias_match(self):
        path = self._gguf_file()
        recs = toks.llama_parse_models(LLAMA_MODELS)
        toks.llama_enrich(recs, {"model_path": path})     # no alias
        self.assertEqual(recs[0].fmt, "gguf")

    def test_no_alias_match_with_several_records_is_noop(self):
        path = self._gguf_file()
        recs = [toks.ModelRecord("llama", "a"), toks.ModelRecord("llama", "b")]
        toks.llama_enrich(recs, {"model_alias": "c", "model_path": path})
        self.assertTrue(all(r.size_bytes is None for r in recs))

    def test_missing_path_and_file_and_props_are_noops(self):
        recs = toks.llama_parse_models(LLAMA_MODELS)
        toks.llama_enrich(recs, {})                              # no model_path
        toks.llama_enrich(recs, "nope")                         # not a dict
        toks.llama_enrich(recs, {"model_alias": "acme/demo-31B-it-GGUF",
                                 "model_path": "/no/such/file.gguf"})
        toks.llama_enrich(recs, {"model_alias": "acme/demo-31B-it-GGUF",
                                 "model_path": __file__})        # exists, not .gguf
        self.assertEqual(recs[0].quant, "-")               # never enriched


class LlamaWiringTests(unittest.TestCase):
    def test_select_providers_llama(self):
        self.assertEqual([p.name for p in toks.select_providers("llama")],
                         ["llama"])

    def test_provider_flag_accepts_llama(self):
        self.assertEqual(toks.parse_args(["--provider", "llama"]).provider, "llama")

    def test_default_url_and_cache_prefix(self):
        self.assertIn("llama:", toks._PROVIDER_PREFIXES)
        with mock.patch.dict(toks.os.environ, {}, clear=True):
            self.assertEqual(toks.llama_url(), "http://127.0.0.1:11435")

    def test_url_env_override(self):
        with mock.patch.dict(toks.os.environ, {"LLAMA_URL": "host:9999"},
                             clear=True):
            self.assertEqual(toks.llama_url(), "http://host:9999")

    def test_headers_from_api_key(self):
        with mock.patch.dict(toks.os.environ, {"LLAMA_API_KEY": "sk-zzz"},
                             clear=True):
            self.assertEqual(toks.llama_headers(),
                             {"Authorization": "Bearer sk-zzz"})
        with mock.patch.dict(toks.os.environ, {}, clear=True):
            self.assertEqual(toks.llama_headers(), {})

    def test_cache_key_and_migration_idempotent(self):
        rec = toks.ModelRecord("llama", "acme/demo-31B-it-GGUF")
        key = toks.cache_key(rec)
        self.assertEqual(key, "llama:acme/demo-31B-it-GGUF")
        cache = {key: {"provider": "llama", "tokens_per_second": 12.3}}
        self.assertEqual(toks.migrate_cache(cache), cache)   # already namespaced


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


# ---- Part B: speculative-decoding acceptance -------------------------------


class DraftStatsFromSseTests(unittest.TestCase):
    CHUNKS = [{"choices": [{"text": "a"}]}, {"choices": [{"text": "b"}]}]

    def test_bench_from_sse_lifts_draft_when_timings_present(self):
        # A llama.cpp impostor squatting on the mlx port carries a timings block.
        final = {"choices": [], "usage": {"completion_tokens": 6},
                 "timings": {"predicted_per_second": 99.0, "draft_n": 10,
                             "draft_n_accepted": 9, "cache_n": 0}}
        lines = _sse(*self.CHUNKS, final)
        r = toks.bench_from_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]))
        self.assertEqual((r.draft_n, r.draft_n_accepted, r.cache_n), (10, 9, 0))

    def test_bench_from_sse_pure_client_stream_has_no_draft(self):
        lines = _sse(*self.CHUNKS,
                     {"choices": [], "usage": {"completion_tokens": 6}})
        r = toks.bench_from_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]))
        self.assertIsNone(r.draft_n)
        self.assertIsNone(r.draft_n_accepted)
        self.assertIsNone(r.cache_n)

    def test_bench_llamacpp_sse_surfaces_draft_from_server_timings(self):
        final = {"choices": [], "usage": {"completion_tokens": 50},
                 "timings": {"predicted_per_second": 42.0, "prompt_ms": 200.0,
                             "draft_n": 48, "draft_n_accepted": 46, "cache_n": 2}}
        lines = _sse(*self.CHUNKS, final)
        r = toks.bench_llamacpp_sse(lines, start=0.0, clock=FakeClock([1.0, 2.0]))
        self.assertEqual((r.draft_n, r.draft_n_accepted, r.cache_n), (48, 46, 2))


class CachedAcceptanceTests(unittest.TestCase):
    def test_ratio_from_stored_pair(self):
        cache = {"unsloth:m": {"draft_n": 48, "draft_n_accepted": 46}}
        self.assertAlmostEqual(
            toks.cached_acceptance(cache, toks.ModelRecord("unsloth", "m")), 46 / 48)

    def test_none_when_draft_zero(self):
        cache = {"unsloth:m": {"draft_n": 0, "draft_n_accepted": 0}}
        self.assertIsNone(
            toks.cached_acceptance(cache, toks.ModelRecord("unsloth", "m")))

    def test_none_when_missing(self):
        cache = {"ollama:m": {"tokens_per_second": 100.0}}
        self.assertIsNone(
            toks.cached_acceptance(cache, toks.ModelRecord("ollama", "m")))


class AccColumnTests(unittest.TestCase):
    def test_header_and_alignment_include_acc(self):
        self.assertEqual(toks.HEADER.index("ACC%"),
                         toks.HEADER.index("TOKENS/S") + 1)
        self.assertIn("ACC%", toks.RIGHT_ALIGN)

    def test_row_renders_acceptance_percent(self):
        cache = {"unsloth:m": {"tokens_per_second": 172.0,
                               "draft_n": 48, "draft_n_accepted": 46}}
        rows = toks.build_rows([toks.ModelRecord("unsloth", "m")], cache)
        self.assertEqual(dict(zip(rows[0], rows[1]))["ACC%"], "96%")

    def test_row_dash_when_no_acceptance(self):
        cache = {"ollama:m": {"tokens_per_second": 100.0}}
        rows = toks.build_rows([toks.ModelRecord("ollama", "m")], cache)
        self.assertEqual(dict(zip(rows[0], rows[1]))["ACC%"], "-")


class CacheNGuardTests(unittest.TestCase):
    class GuardProvider:
        def __init__(self, cache_n):
            self.host = "http://127.0.0.1:8888"
            self.expected_backend = None
            self._cache_n = cache_n

        def benchmark(self, rec, prompt, max_tokens):
            return toks.BenchResult(150.0, 0.1, tps_source="llamacpp_timings",
                                    observed_backend="llamacpp", draft_n=48,
                                    draft_n_accepted=46, cache_n=self._cache_n)

    def _run(self, cache_n):
        rec = toks.ModelRecord("unsloth", "m", benchmarkable=True)
        cache = {}
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            toks.run_benchmarks([rec], {"unsloth": self.GuardProvider(cache_n)},
                                cache, samples=1)
        return cache[toks.cache_key(rec)], err.getvalue()

    def test_warns_when_cache_n_exceeds_budget(self):
        entry, err = self._run(64)
        self.assertIn("cache_n", err)
        self.assertIn("64", err)
        self.assertAlmostEqual(entry["tokens_per_second"], 150.0)   # recorded anyway

    def test_no_warning_at_cache_n_zero(self):
        _, err = self._run(0)
        self.assertNotIn("cache_n", err)


# ---- Part C: content spread across prompt kinds (--bench-suite) -------------


class BenchSuiteTests(unittest.TestCase):
    class SuiteProvider:
        host = "http://127.0.0.1:8888"
        expected_backend = None

        def __init__(self):
            self.prompts = []

        def benchmark(self, rec, prompt, max_tokens):
            self.prompts.append(prompt)
            if prompt.endswith(toks.BENCH_PROMPTS["code"]):
                return toks.BenchResult(300.0, 0.1, observed_backend="llamacpp",
                                        draft_n=50, draft_n_accepted=48)
            return toks.BenchResult(100.0, 0.2, observed_backend="llamacpp",
                                    draft_n=10, draft_n_accepted=3)

    def test_bench_prompts_has_prose_and_code(self):
        self.assertEqual(toks.BENCH_PROMPTS["prose"], toks.DEFAULT_PROMPT)
        self.assertIn("code", toks.BENCH_PROMPTS)

    def test_bench_suite_flag_default_false(self):
        self.assertFalse(toks.parse_args([]).bench_suite)
        self.assertTrue(toks.parse_args(["--bench-suite"]).bench_suite)

    def test_suite_writes_by_kind_with_prose_top_level(self):
        rec = toks.ModelRecord("unsloth", "m", benchmarkable=True)
        provider = self.SuiteProvider()
        cache = {}
        with contextlib.redirect_stderr(io.StringIO()):
            toks.run_benchmarks([rec], {"unsloth": provider}, cache, samples=1,
                                kinds=["prose", "code"])
        entry = cache["unsloth:m"]
        self.assertAlmostEqual(entry["tokens_per_second"], 100.0)     # prose
        self.assertEqual(entry["draft_n"], 10)                        # prose pooled
        self.assertIn("by_kind", entry)
        self.assertAlmostEqual(entry["by_kind"]["prose"]["tokens_per_second"], 100.0)
        self.assertAlmostEqual(entry["by_kind"]["code"]["tokens_per_second"], 300.0)
        self.assertEqual(entry["by_kind"]["code"]["draft_n"], 50)
        self.assertEqual(entry["by_kind"]["code"]["draft_n_accepted"], 48)

    def test_default_run_writes_no_by_kind(self):
        rec = toks.ModelRecord("ollama", "m", digest="d", benchmarkable=True)
        provider = self.SuiteProvider()
        cache = {}
        with contextlib.redirect_stderr(io.StringIO()):
            toks.run_benchmarks([rec], {"ollama": provider}, cache, samples=1)
        self.assertNotIn("by_kind", cache["ollama:d"])
        self.assertEqual(len(provider.prompts), 1)                    # prose only


class CodeColumnTests(unittest.TestCase):
    def _suite_cache(self):
        return {"unsloth:m": {"tokens_per_second": 100.0, "draft_n": 10,
                              "draft_n_accepted": 3,
                              "by_kind": {"code": {"tokens_per_second": 300.0,
                                                   "draft_n": 50,
                                                   "draft_n_accepted": 48}}}}

    def test_code_columns_present_when_by_kind_code(self):
        rows = toks.build_rows([toks.ModelRecord("unsloth", "m")], self._suite_cache())
        header = rows[0]
        self.assertEqual(header.index("CODE/S"), header.index("ACC%") + 1)
        self.assertEqual(header.index("CODE%"), header.index("CODE/S") + 1)
        cell = dict(zip(header, rows[1]))
        self.assertEqual(cell["CODE/S"], "300.0")
        self.assertEqual(cell["CODE%"], "96%")

    def test_code_columns_absent_without_by_kind(self):
        cache = {"ollama:m": {"tokens_per_second": 100.0}}
        rows = toks.build_rows([toks.ModelRecord("ollama", "m")], cache)
        self.assertNotIn("CODE/S", rows[0])
        self.assertNotIn("CODE%", rows[0])

    def test_legacy_entry_renders_without_samples_draft_or_by_kind(self):
        cache = {"ollama:d": {"tokens_per_second": 80.0, "provider": "ollama"}}
        rec = toks.ModelRecord("ollama", "old", digest="d")
        self.assertAlmostEqual(toks.cached_tps(cache, rec), 80.0)
        self.assertIsNone(toks.cached_acceptance(cache, rec))
        rows = toks.build_rows([rec], cache)
        self.assertNotIn("CODE/S", rows[0])                          # no by_kind
        cell = dict(zip(rows[0], rows[1]))
        self.assertEqual(cell["TOKENS/S"], "80.0")
        self.assertEqual(cell["ACC%"], "-")


if __name__ == "__main__":
    unittest.main()
