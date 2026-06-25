# toks: Unsloth local listing + observed-backend identity in bench data — design

Date: 2026-06-25

## Summary

Two related fixes, both surfaced while debugging "Unsloth says reachable but no
models appear":

1. **Unsloth listing source.** The `unsloth` provider lists models via the
   OpenAI-compatible `GET /v1/models`, which on Unsloth Studio returns only the
   models *currently loaded for serving* — empty when nothing is loaded. So the
   provider probes reachable (`unsloth ✓`) yet contributes zero rows even though
   the machine holds many downloaded Unsloth models. Switch the listing to
   Studio's local-inventory endpoint so downloaded models appear.

2. **Observed backend recorded in bench data.** The benchmark cache stamps
   `provider` = the *configured label* (`mlx`, `unsloth`, …) — i.e. "whatever
   answered at this URL". Nothing verifies which engine actually served the
   request. A default port like mlx's `:8080` is commonly reused; if a different
   server binds it, toks benchmarks the impostor and writes the number under
   `mlx:` forever, silently corrupting cross-runtime comparisons. Capture the
   *observed* backend at bench time and persist it alongside the throughput.

## Motivation: cross-runtime comparison is the point

The same physical weights on disk perform differently depending on the serving
runtime. `mlx-community/Qwen3-Coder-Next-8bit` in the HF cache can be served by
`mlx_lm` (Apple MLX) *and* by Unsloth Studio (transformers/llama.cpp). Listing it
under both providers — same model name, different `TOKENS/S` — is not
double-counting; it is exactly the A/B worth measuring. This reframes the
filtering: overlap with the other providers is **intentional**, so the Unsloth
listing is *not* namespace-scoped to `unsloth/*`.

For that comparison to be trustworthy, each cached number must record which engine
produced it (fix 2).

## Current state (relevant pieces)

- `UnslothProvider.list_models` (toks:1259) → `GET /v1/models` →
  `unsloth_parse_models` (toks:698, ids-only) → `unsloth_enrich` (toks:725, reads
  the cached GGUF header from disk). Benchmark POSTs `/v1/completions` (unchanged).
- `BenchResult` (toks:67) carries only `tokens_per_second` + `time_to_first_token`.
- Five bench parsers each key off a *server-determined response shape*:
  - `parse_ollama_bench` (toks:159) — `eval_count`/`eval_duration` (Ollama)
  - `parse_lmstudio_bench` (toks:145) — `stats.tokens_per_second` (LM Studio)
  - `parse_llamacpp_timings` (toks:764) — `timings.predicted_per_second` (llama.cpp)
  - `bench_from_sse` (toks:526) — no server stats, client-timed (mlx_lm)
  - `bench_unsloth_sse` (toks:784) — llama.cpp timings, else client-timed fallback
- `run_benchmarks` (toks:1385) writes the cache entry (`provider`, `model`, tps,
  ttft, `benchmarked_at`, prompt, max_tokens). `cache_key` (toks:857) =
  `{provider}:{digest-or-name}`.

## Verified endpoint behaviour (Unsloth Studio, authed)

- `GET /v1/models` → `{"object":"list","data":[]}` (loaded-only; empty now).
- `GET /api/hub/local` → every **downloaded** model across sources, each with:
  `id`, `source` (`hf_cache` | `lmstudio` | ollama-manifest id), `model_format`
  (`gguf`/`safetensors`/`unknown`), `runtime`, `size_bytes`, and
  `capabilities.can_chat`.
- `can_chat` is **imperfect**: it marks `unsloth/bge-small-en-v1.5-GGUF`
  (embedding) and `kyutai/*` (speech) as `True`, and a real chat model
  (`unsloth/gemma-4-26B-A4B-it-GGUF`) was elsewhere mislabeled — so it cannot be
  the sole text-generation filter.

## Design

### Part A — list Unsloth from local inventory

`UnslothProvider.list_models`: `GET /api/hub/local` (auth) →
`unsloth_parse_local(payload)` → `unsloth_enrich` (unchanged; HF-cache, local-only).

`unsloth_parse_local(payload)` builds a `ModelRecord` (`provider="unsloth"`,
`benchmarkable=True`, `raw=item`) for each entry passing **all** of:

- `source == "hf_cache"` — scope to the shared HF cache. Drops `lmstudio`-dir and
  ollama-manifest entries, which toks' own `lmstudio`/`ollama` providers list
  natively (avoids listing the *same server's* models twice). The HF cache is the
  shared store where Unsloth-served GGUF **and** MLX/safetensors weights live, so
  this stays format-agnostic — a future `unsloth/…-MLX-8bit` appears automatically.
- `capabilities.can_chat is True` — drops image gen (`Tongyi-MAI/Z-Image-Turbo`).
- `model_format in ("gguf", "safetensors")` — drops `unknown`/partial junk
  (`ggml-org/models`, `kyutai/tts-voices`).
- id not matching `NON_TEXTGEN_HINTS` (case-insensitive substrings:
  `bge`, `embed`, `rerank`, `tts`, `stt`, `whisper`, `asr`) — best-effort backstop
  for the non-text models `can_chat` mislabels (embeddings, speech).

The entry's `id` is an HF repo id (e.g. `unsloth/gemma-4-31B-it-GGUF`), feeding
`unsloth_enrich` → `_mlx_snapshot` exactly as today. `model_format`/`size_bytes`
from the entry are advisory only; disk enrichment remains authoritative.

**No namespace filter** — `mlx-community/*` chat models in the HF cache are listed
under `unsloth` *on purpose*, so they can be benchmarked under Studio and compared
against the `mlx` provider's number for the identical files.

**Limits (documented, accepted):** text-gen classification is best-effort (Studio's
flags are incomplete); the filter may occasionally include a model that then fails
to benchmark (graceful — a warning, no cache entry) or exclude a mislabeled one.
Filtering is meaningful only for a local Studio (HF cache readable), consistent
with the existing local-only enrichment.

`unsloth_parse_models` (the `/v1/models` parser) is removed along with its two
tests; the provider no longer calls it.

### Part B — record the observed backend in bench data

**`BenchResult` gains three fields** (all optional, defaulted):

```python
tps_source: str | None = None        # measurement path actually used
observed_backend: str | None = None  # engine inferred from the evidence
system_fingerprint: str | None = None
```

`tps_source` ∈ {`ollama`, `lmstudio_stats`, `llamacpp_timings`, `client_timed`}.
`observed_backend` ∈ {`ollama`, `lmstudio`, `llamacpp`, `mlx_lm`, `unknown`}.

**Each parser stamps what it saw:**

- `parse_ollama_bench` → source `ollama`, backend `ollama`.
- `parse_lmstudio_bench` → source `lmstudio_stats`, backend `lmstudio`; lift
  `system_fingerprint` if present.
- `parse_llamacpp_timings` → source `llamacpp_timings`, backend `llamacpp`.
- `bench_from_sse` (mlx) → source `client_timed`. **Crucially still inspects each
  event** for a `timings` block and for `system_fingerprint`: a llama.cpp build
  fingerprint (`^b\d+-`) or a `timings` block under the mlx provider means the
  port is actually a llama.cpp server → backend `llamacpp`; otherwise `mlx_lm`.
  This is what catches an impostor on `:8080` even though we client-time.
- `bench_unsloth_sse` → `llamacpp` when timings present, else the `bench_from_sse`
  fallback's classification (Studio is multi-runtime; either is legitimate).

**Provider declares its expected backend** (class attr `expected_backend`):
ollama→`ollama`, lmstudio→`lmstudio`, mlx→`mlx_lm`, unsloth→`None` (multi-runtime;
record, don't flag).

**`run_benchmarks` mismatch guard + persistence.** After a successful benchmark:

- If `provider.expected_backend` is set and `result.observed_backend` not in
  {`expected`, `unknown`}: print a loud warning — e.g.
  `warning: {model}: expected {expected} at {host} but observed {observed}
  (fingerprint={fp}); recording the observed backend`.
- Write into the cache entry, alongside the existing fields:
  `endpoint` = `provider.host`, `tps_source`, `observed_backend`,
  `system_fingerprint`.

**Cache key unchanged** (`{provider}:{model}`): identity lives in the *value*, not
the key, so history isn't fragmented and a re-bench whose `observed_backend`/
`system_fingerprint` differs from the stored one is visible (and warns). Existing
entries lacking the fields read as backend `unknown` — fully back-compat;
`migrate_cache` needs no change.

## Non-goals

- Collapsing the four providers into Studio's aggregator (rejected earlier:
  kills per-runtime benchmarking, makes Studio a single point of failure).
- Namespace-scoping Unsloth to `unsloth/*` (rejected: deletes the cross-runtime
  comparison).
- Arch-level text-gen detection from GGUF/safetensors headers (future refinement;
  the name-hint backstop covers the present cases).
- Re-pointing or fixing the `mlx` provider's default `:8080`. The bench guard now
  *reports* when `:8080` is not `mlx_lm`; pointing it at a real `mlx_lm.server` is
  the user's runtime concern, not a code change here.
- Grouping the table by model so a model's per-runtime rows sit adjacent (nice for
  comparison, but a separate rendering change).

## Test plan (network-free, `unittest`)

Part A — `unsloth_parse_local`:
- keeps `source==hf_cache` + `can_chat` + gguf/safetensors; one record each.
- drops `source==lmstudio` and ollama-manifest entries.
- drops `can_chat==False` (image) and `model_format=="unknown"`.
- drops `NON_TEXTGEN_HINTS` matches (`bge`, `tts`, `stt`) despite `can_chat==True`.
- keeps non-`unsloth/*` chat models (e.g. `mlx-community/*`) — overlap by design.
- empty/malformed payload → `[]`.
- `UnslothProvider.list_models` calls `/api/hub/local` and enriches when local
  (mock `http_json`).

Part B — observed backend:
- each parser sets the expected `tps_source`/`observed_backend`
  (+ `system_fingerprint` where supplied).
- `bench_from_sse` flags `llamacpp` when a stream carries a `timings` block or a
  `b\d+-…` fingerprint; `mlx_lm` otherwise.
- `run_benchmarks` writes `endpoint`/`tps_source`/`observed_backend`/
  `system_fingerprint` into the entry and warns on mismatch (capture stderr);
  no warning when observed matches or provider expects `None`.
- legacy entries without the fields still load/migrate unchanged.
