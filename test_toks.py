"""Network-free unit tests for toks.

The tool is an extensionless executable, so load it as a module by path.
"""
import importlib.machinery
import importlib.util
import pathlib
import unittest

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
        self.assertEqual(cell["CTX"], "8192")          # loaded ctx preferred
        self.assertEqual(cell["MODIFIED"], "-")

    def test_ollama_row_uncached_shows_dashes(self):
        rows = toks.build_rows(self.records, self.cache)
        cell = dict(zip(rows[0], rows[2]))             # scribe:8b, not cached
        self.assertEqual(cell["TOKENS/S"], "-")
        self.assertEqual(cell["TTFT"], "-")
        self.assertEqual(cell["CTX"], "40960")         # max ctx fallback
        self.assertEqual(cell["SIZE"], "4.4 GB")

    def test_table_renders_aligned_string(self):
        rows = toks.build_rows(self.records, self.cache)
        out = toks.table(rows)
        self.assertIn("PROVIDER", out.splitlines()[0])
        self.assertEqual(len(out.splitlines()), 3)


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


if __name__ == "__main__":
    unittest.main()
