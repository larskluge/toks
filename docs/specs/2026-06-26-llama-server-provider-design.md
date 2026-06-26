# toks: `llama` provider (llama.cpp server) — design

Date: 2026-06-26

## Summary

Add a fifth backend to `toks`: **`llama`**, an OpenAI-compatible **llama.cpp
server** (`llama-server`), default `http://127.0.0.1:11435`. It is listed from
`/v1/models` (which already carries param count, on-disk size, and context
length in a `meta` block), benchmarked from the server's own llama.cpp `timings`
(accurate, server-measured — the existing path), and enriched with quant / MoE /
exact-param metadata by reading the loaded model's GGUF header, whose absolute
path is reported by `/props` (`model_path`).

The probed `:11435` server on this machine appears to be the llama.cpp server
that **Unsloth Studio** itself spawns (its `/props` carries an Unsloth-vendored
chat template and a `model_path` in the HF cache; the existing `unsloth` provider
talks to Studio's *management* API at `:8888`). Listing the same weights under
both `unsloth` and `llama` is **intentional and kept** — it lets the exact same
files be compared across the two entry points, matching the cross-runtime
comparison philosophy already stated in `unsloth_parse_local`'s docstring.

## Goals

- Add a `llama` provider: list, benchmark (accurate, server-measured), and enrich
  GGUF metadata (quant / MoE / exact params / ctx) from the loaded model's GGUF.
- Populate every existing column for `llama` rows where the data is available —
  `SIZE`, `PARAMS`, `BPW`, `CTX` come straight from the `/v1/models` `meta` block
  even over a network; `BPW`/`TAG` quant detail comes from the local GGUF read.
- Reuse the existing llama.cpp benchmark machinery (`parse_llamacpp_timings` and
  the streaming SSE timer) rather than duplicating it.
- Keep the single-file, standard-library-only constraint and the existing
  per-provider class + cache-namespace pattern.

## Non-goals / out of scope

- De-duplicating the `unsloth`/`llama` overlap. Both list separately by design.
- Re-pointing or changing the `unsloth` (`:8888`) or `mlx` (`:8080`) providers.
- Reading GGUF metadata over the network. Enrichment reads the absolute
  `model_path` from `/props`; off-box that path won't exist and enrichment is a
  silent no-op (same locality contract as `mlx`/`unsloth`).
- Multi-model `llama-server` mapping. `llama-server` runs one model per process
  (`total_slots` aside); `/v1/models` lists it and `/props` names it. If a build
  ever lists several, we still list them all from `meta`, and GGUF-enrich only the
  one `/props` reports loaded (see "Enrichment" for the matching rule).
- Any column, header, or table-layout change. The change is purely additive (one
  new provider; the `PROVIDER` column already renders any label).

## Current state (relevant pieces)

- Providers are plain classes with `name` + `expected_backend`, selected in
  `select_providers` (toks:1415); each implements `list_models()` and
  `benchmark()`. `gather_records` (toks:1427) isolates per-provider failures.
- `UnslothProvider` (toks:1370) is described in-code as "a llama.cpp server": it
  warms up, streams `/v1/completions`, and reads server `timings` via
  `bench_unsloth_sse` (toks:891) → `parse_llamacpp_timings` (toks:869).
- `read_gguf_meta` (toks:655) parses a GGUF header for exact param count, quant
  label, context length, and MoE expert counts. `_mlx_snapshot`/`select_main_gguf`
  locate weights in the HF cache; `unsloth_enrich` (toks:816) is the existing
  "read the GGUF, backfill the record" routine.
- `_classify_streamed_backend` (toks:85) infers the engine from a `timings` block
  or a `b<build>-…` fingerprint; `run_benchmarks` (toks:1507) warns when the
  `observed_backend` contradicts a provider's `expected_backend`.
- Per-provider env URLs: `ollama_url`/`lmstudio_url`/`mlx_url`/`unsloth_url`
  (toks:1117+) wrap `normalize_host`. Cache keys are namespaced via
  `_PROVIDER_PREFIXES` (toks:969); `migrate_cache` treats any key with a known
  prefix as already-migrated.
- `parse_args` (toks:1555) gates `--provider` to a fixed `choices` tuple.

### Observed API shape (probed `:11435`)

`GET /v1/models` returns both an Ollama-style `models[]` and an OpenAI-style
`data[]`. We use `data[]`:

```json
{"data": [{"id": "unsloth/gemma-4-31B-it-qat-GGUF", "owned_by": "llamacpp",
  "meta": {"n_ctx": 131072, "n_ctx_train": 262144,
           "n_params": 30697345596, "size": 17271834864}}]}
```

`GET /props` returns `model_alias` and the exact GGUF path:

```json
{"model_alias": "unsloth/gemma-4-31B-it-qat-GGUF",
 "model_path": ".../snapshots/<sha>/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
 "n_ctx": 131072, "total_slots": 1}
```

`GET /health` → `{"status":"ok"}`. Quant is **not** in `meta` — only the GGUF read
yields it, so the GGUF enrichment is what populates `TAG`'s quant and `BPW`.

## Design

### 1. Config / wiring

- `DEFAULT_LLAMA_URL = "http://127.0.0.1:11435"` (the port the user runs; note
  `mlx` already defaults to llama.cpp's own `:8080`, so `11435` is the right
  default here and avoids collision).
- `llama_url()` → `normalize_host(os.environ.get("LLAMA_URL"), DEFAULT_LLAMA_URL)`.
- `llama_headers()` → `Bearer` from `LLAMA_API_KEY` if set (vanilla `llama-server`
  supports `--api-key`); empty dict otherwise. Mirrors `unsloth_headers`.
- `select_providers`: add `"llama": LlamaProvider` (appended after `unsloth`).
- `parse_args` `choices`: add `"llama"`.
- `_PROVIDER_PREFIXES`: add `"llama:"`.

  *Migration edge (accepted):* a legacy, pre-namespace Ollama cache key of the
  literal form `llama:<x>` would now be read as already-migrated instead of
  re-prefixed to `ollama:llama:<x>`. Legacy keys were Ollama digests (hex) or
  `name:tag` tags (e.g. `llama3:8b`, which does **not** start with `llama:`), so
  the real-world collision risk is negligible. Documented, not guarded.

### 2. `LlamaProvider`

```python
class LlamaProvider:
    name = "llama"
    expected_backend = "llamacpp"   # mislabel detection: warn if something else
                                    # answers at this URL

    def __init__(self):
        self.host = llama_url()
        self.headers = llama_headers()

    def list_models(self):
        data = http_json(f"{self.host}/v1/models", headers=self.headers,
                         timeout=LIST_TIMEOUT)
        records = llama_parse_models(data)
        # meta gives size/params/ctx even remotely; the quant (BPW/TAG) lives only
        # in the GGUF, readable when the server runs on this machine.
        if records and is_local_host(self.host):
            try:
                props = http_json(f"{self.host}/props", headers=self.headers,
                                  timeout=INFO_TIMEOUT)
            except RuntimeError:
                props = {}
            llama_enrich(records, props)
        return records

    def benchmark(self, rec, prompt, max_tokens):
        # identical machinery to unsloth/mlx: 1-token warmup, then stream and read
        # the server's llama.cpp timings (auth token rides on every request via
        # http_sse's headers; the model is already loaded so warmup is cheap).
        ...
        lines = http_sse(f"{self.host}/v1/completions", payload, BENCH_TIMEOUT,
                         headers=self.headers)
        return bench_llamacpp_sse(lines, start, otherwise="llamacpp")
```

### 3. `llama_parse_models(payload)` (pure)

Parse `data[]` → `ModelRecord`s:

- `name` = `item["id"]`.
- `fmt = "gguf"` — `llama-server` serves GGUF only, so this is always correct and
  gives a sensible `TAG` even without local enrichment.
- From `item["meta"]` (best-effort, each guarded):
  - `param_count = meta["n_params"]`, `param_estimated = False` (server-exact),
    and `params = human_params(n_params, None, None)` so a useful string shows
    even off-box. GGUF enrichment later refines this with MoE-aware detail.
  - `size_bytes = meta["size"]`.
  - `ctx_loaded = meta["n_ctx"]` (running context), `ctx_max = meta["n_ctx_train"]`
    (trained max). `build_rows` renders `ctx_loaded or ctx_max`.
- `benchmarkable = True` (a loaded LLM; a failed bench only warns, per the `mlx`
  optimism convention).
- `provider = "llama"`, `raw = item`.
- Missing/non-list `data` → `[]`. Malformed `meta` leaves those fields at default.

### 4. `llama_enrich(records, props)` (pure)

`props` is the parsed `/props` body. Reuse the existing GGUF-read/backfill logic:

- Read `model_path` (str) and `model_alias` from `props`; if no `model_path`, no-op.
- Pick the target record: the one whose `name == model_alias`, else — if there's
  exactly one record — that record (the single-model common case). If neither
  matches, no-op (don't guess across several listed models).
- If `Path(model_path)` exists and is a `*.gguf`, `read_gguf_meta(model_path)` and
  backfill `fmt="gguf"`, `quant`, `ctx_max`, `moe`, exact `param_count`
  (`param_estimated=False`), and the MoE-aware `params` string — exactly the GGUF
  branch of `unsloth_enrich`. Absent/malformed file → leave the record as parsed.

  To avoid duplicating that branch, factor the GGUF-read-and-backfill body of
  `unsloth_enrich` into a small helper `apply_gguf_meta(rec, gguf_path)` that
  backfills **only** the GGUF-derived fields — `fmt`, `quant`, `ctx_max`, `moe`,
  exact `param_count`/`param_estimated`, the MoE-aware `params` string, and
  `size_bytes` (from the GGUF file stat) — and returns whether it succeeded.
  It does **not** touch `modified_at`; each caller sets that itself, so behavior is
  unchanged:
  - `unsloth_enrich`: keeps its HF-cache `model_path` discovery and its existing
    `_snapshot_modified_at(rec, ref)` (mtime of `refs/main`), then calls the helper.
  - `llama_enrich`: gets the path straight from `/props`, calls the helper, then
    sets `modified_at` from the GGUF file's own mtime (no HF `refs/main` to read).
  For `llama`, the GGUF file size equals the `meta["size"]` already set, so the
  helper's `size_bytes` write is consistent, not a conflict.

### 5. Benchmark: generalize the SSE timer

`bench_unsloth_sse` already does exactly what `llama` needs (prefer server
`timings`, fall back to client timing). It is now used by two llama.cpp consumers,
so **rename it `bench_llamacpp_sse`** (the honest name — it parses llama.cpp
`timings`) and add a parameter:

```python
def bench_llamacpp_sse(lines, start, clock=time.monotonic, otherwise="unknown"):
    ...
    observed_backend=_classify_streamed_backend(fingerprint, False, otherwise=otherwise)
```

- `UnslothProvider` calls it with the default `otherwise="unknown"` (Studio is
  multi-runtime) — **behavior unchanged**.
- `LlamaProvider` calls it with `otherwise="llamacpp"` (a pure llama.cpp server;
  in practice `timings` are always present so this fallback is unreachable, but it
  keeps the semantics honest). `expected_backend="llamacpp"` + `timings` →
  `observed="llamacpp"` → no false mislabel warning.

`http_sse` already accepts `headers`, so the auth token rides along.

*Alternative considered:* leave `bench_unsloth_sse` named as-is and call it from
`LlamaProvider`. Rejected — a `llama` provider calling a `bench_unsloth_*`
function misleads future readers, and the rename is a one-line-per-call-site
change. This is the only rename; no other behavior of `unsloth` changes.

## Testing (TDD)

Mirror the existing per-provider test classes; reuse the `_build_gguf` /
`_sse` / `FakeClock` helpers already in `test_toks.py`.

- **`LlamaListingTests`** — `llama_parse_models` on an anonymized `/v1/models`
  capture: records carry `provider="llama"`, `fmt="gguf"`, exact `param_count`,
  `size_bytes`, `ctx_loaded`/`ctx_max` from `meta`, `benchmarkable=True`;
  missing/empty `data` → `[]`; malformed `meta` leaves dashes.
- **`LlamaEnrichmentTests`** — `llama_enrich` with a temp `_build_gguf` file and a
  `props` dict pointing `model_path` at it: backfills quant/ctx/MoE/exact params;
  alias-matches the right record; single-record fallback; no `model_path`, missing
  file, and unmatched alias are all no-ops (never raise).
- **`bench_llamacpp_sse`** — rename the existing `BenchUnslothSseTests` calls; add
  a case asserting `otherwise="llamacpp"` is used when no timings/fingerprint
  appear, and that `otherwise` defaults to `"unknown"` (the unsloth path). Update
  the `ObservedBackendTests.test_unsloth_fallback_backend_is_unknown` call site.
- **`LlamaWiringTests`** — `select_providers("llama")`; `select_providers("all")`
  includes `llama`; `parse_args(["--provider","llama"])`; `llama_url()` default and
  `LLAMA_URL` override; `llama_headers()` with/without `LLAMA_API_KEY`;
  `"llama:"` in `_PROVIDER_PREFIXES`; `cache_key`/`migrate_cache` idempotent for a
  `llama:` key.

All 147 existing tests must still pass (only the `bench_unsloth_sse` → 
`bench_llamacpp_sse` call sites change).

## Acceptance criterion (live)

Beyond unit tests, the feature must work end-to-end against the running
`:11435` server:

- `./toks --provider llama` lists `unsloth/gemma-4-31B-it-qat-GGUF` with a
  populated `SIZE`, `PARAMS`, `BPW` (quant from the GGUF), and `CTX`.
- `./toks --provider llama --bench` (benches uncached models, incl. gemma) prints
  a real `TOKENS/S` and `TTFT` for the row, sourced from the server's llama.cpp
  `timings` and cached under a `llama:` key. A re-run of `./toks` shows the cached
  value persists.

(`--bench` with an explicit target matches on the exact model `id`, i.e.
`--bench unsloth/gemma-4-31B-it-qat-GGUF`, not the bare `gemma-…` substring.)

## Docs

- Update the module docstring and `parse_args` description: backends become
  "Ollama, LM Studio, mlx-lm, Unsloth Studio, and llama.cpp server".
- Add a short `llama` row to `README.md`'s provider/env table (`LLAMA_URL`,
  `LLAMA_API_KEY`, default `:11435`).
- This spec lives at `docs/specs/2026-06-26-llama-server-provider-design.md`.
