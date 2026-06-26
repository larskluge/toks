# toks: representative, cache-safe benchmarks + speculative-decoding acceptance — design

Date: 2026-06-27

## Summary

Five fixes to the benchmark path, all surfaced while A/B-ing the *same* weights
under Unsloth Studio (llama.cpp + MTP) vs `mlx_lm` and ollama. The current method —
one fixed prompt, 128 tokens, a single timed run (`run_benchmarks`, toks:1731) —
produces a number that is neither representative nor reproducible once a backend
uses **speculative decoding** (Unsloth Studio defaults to `ngram-mod,draft-mtp`).
The same model measured anywhere from **~100 to ~1000 tok/s** depending on prompt
content and whether a prior identical prompt was cached. A single sample of that
distribution, stamped into the cache as *the* throughput, silently corrupts the
cross-runtime comparison toks exists to make.

1. **Per-run prompt nonce.** Re-running a benchmark with the same deterministic
   (temp-0) prompt lets the server's ngram cache + KV prefix cache memorise the
   prior output, inflating throughput up to ~6×. Prefix each timed prompt with a
   random nonce so no two runs share a prefix — measurements become independent.
2. **N-sample median + spread.** A single run varied ±20–30% even ignoring
   content. Take the median of N independent (nonced) samples; record the spread.
3. **Surface speculative-decoding acceptance.** The llama.cpp `timings` block toks
   already parses carries `draft_n` / `draft_n_accepted`. That ratio *is* the
   explanation for the variance. Capture it and show an `ACC%` column.
4. **Longer generation.** `DEFAULT_MAX_TOKENS` 128 → 256: short runs over-weight
   warmup/first-chunk effects and never reach steady state.
5. **Content spread (opt-in).** Spec-decode throughput differs ~2× between prose
   and code (acceptance is content-dependent). An opt-in `--bench-suite` measures a
   `{prose, code}` pair so the cached number isn't a single content type's luck.

Fixes 1–4 are default-on, additive, and small. Fix 5 is opt-in and is the only one
that touches table/cache shape; it ships second (see **Phasing**).

## Motivation: a representative number is as load-bearing as the right backend

The 2026-06-25 spec established that a cached throughput is only trustworthy if it
records *which engine* produced it (`observed_backend`), because a mislabeled
number poisons the cross-runtime A/B. The same logic extends one step: a number
from a *non-representative or cache-inflated* run poisons the A/B just as badly —
and worse, invisibly, because the row looks normal. When one side speculates
(llama.cpp + MTP) and the other does not (`mlx_lm`), comparing a single fixed-prompt
sample is comparing a coin-flip against a constant. Making the measurement
representative (1, 2, 4, 5) and self-explaining (3) closes that gap.

## Current state (relevant pieces)

- `DEFAULT_PROMPT` (toks:32), `DEFAULT_MAX_TOKENS = 128` (toks:33),
  `BENCH_TIMEOUT = 600` (toks:38).
- `BenchResult` (toks:71): `tokens_per_second`, `time_to_first_token`, `tps_source`,
  `observed_backend`, `system_fingerprint`.
- `parse_llamacpp_timings` (toks:999) reads only `timings.predicted_per_second` and
  `prompt_ms` from a block that *also* contains draft and cache counters.
- `bench_from_sse` (toks:563) client-times and classifies the backend
  (`_classify_streamed_backend`), but extracts no draft stats.
- Five providers each expose `benchmark(self, rec, prompt, max_tokens)`:
  Ollama (toks:1421), LMStudio (toks:1466), Mlx (toks:1492), Unsloth (toks:1561),
  Llama (toks:1613). Each does a 1-token warmup then one timed run.
- `run_benchmarks` (toks:1731) calls `provider.benchmark(rec, DEFAULT_PROMPT,
  DEFAULT_MAX_TOKENS)` **once** and writes the cache entry (`provider`, `model`,
  `tokens_per_second`, `time_to_first_token`, `endpoint`, `tps_source`,
  `observed_backend`, `system_fingerprint`, `benchmarked_at`, `prompt`,
  `max_tokens`).
- `HEADER` (toks:1174), `RIGHT_ALIGN` (toks:1176), `build_rows` (toks:1179),
  `table` (toks:1223). Cache read helpers `cached_tps` (toks:1145),
  `cached_ttft` (toks:1155). `cache_key` (toks:1108) = `{provider}:{digest-or-name}`;
  `migrate_cache` (toks:1119).

## Verified behaviour (Unsloth Studio, authed, 2026-06-27)

Real `/v1/completions` `timings` block on a code prompt:

```json
{"cache_n":1,"prompt_n":4,"prompt_ms":19.8,"predicted_n":64,"predicted_ms":366.7,
 "predicted_per_second":174.5,"draft_n":48,"draft_n_accepted":46}
```

- `draft_n_accepted / draft_n` = 46/48 = **96% acceptance** — the MTP head's hit
  rate on this content. High here (predictable code); far lower on freeform prose.
- `cache_n` = tokens served from the KV prefix cache. Non-zero means the prompt (or
  a prefix) was seen before — the inflation tell.
- **Inflation reproduced:** re-sending the *same* code prompt at temp 0 climbed
  165 → 222 → 653 → ~996 tok/s across runs as `ngram-mod` memorised the (identical)
  output. **Content spread:** across distinct code prompts, acceptance and thus
  throughput ranged ~160–630 tok/s (LRU-cache vs Dijkstra); prose sat ~100–135.
  `mlx_lm`/ollama (no speculation) stayed flat ±2% throughout.

These three numbers — acceptance, cache_n, and the spread — are exactly what the
current single-shot path throws away.

## Design

### Part A — representative, cache-safe sampling (fixes 1, 2, 4)

Keep each provider's `benchmark(rec, prompt, max_tokens)` signature **unchanged**
(5 implementations + their tests stay put). The nonce, the N-sample loop, and
aggregation live in `run_benchmarks` and small pure helpers.

**Nonce (fix 1).** A module helper makes each timed prompt's *prefix* unique:

```python
import secrets
def nonce_prompt(prompt):
    # Unique bytes lead, so no two runs share a KV prefix and ngram-mod can't
    # replay a prior identical generation. Suffix is the real prompt.
    return f"[{secrets.token_hex(3)}] {prompt}"
```

Prefix (not suffix) is deliberate: llama.cpp's prefix cache keys on the leading
tokens, so a leading nonce forces `cache_n == 0` on the timed run. The nonce is ~3
tokens, negligible against 256.

**N samples + median (fix 2).** `DEFAULT_SAMPLES = 3` (constant; overridable via
`--samples N`, default 3). `run_benchmarks` calls `provider.benchmark(rec,
nonce_prompt(base), max_tokens)` N times, collecting `BenchResult`s, then:

```python
def aggregate_samples(results):
    tps   = [r.tokens_per_second for r in results]
    ttfts = [r.time_to_first_token for r in results if r.time_to_first_token]
    dn    = sum(r.draft_n or 0 for r in results)
    da    = sum(r.draft_n_accepted or 0 for r in results)
    return {
        "tokens_per_second": statistics.median(tps),
        "tps_min": min(tps), "tps_max": max(tps), "samples": len(tps),
        "time_to_first_token": statistics.median(ttfts) if ttfts else None,
        "draft_n": dn or None, "draft_n_accepted": (da if dn else None),
    }
```

Acceptance is **pooled** (Σaccepted / Σdrafted), not a mean of per-run ratios — the
statistically correct aggregate. The first sample still pays cold-mmap warmup (each
`benchmark()` warms internally); medians of the remaining samples absorb it without
special-casing.

`stdlib only`: `secrets`, `statistics` are both standard library — no new deps.

**Longer generation (fix 4).** `DEFAULT_MAX_TOKENS` 128 → 256. With N=3 that is
3×256 tokens per model; `BENCH_TIMEOUT` (600 s) already covers JIT loads.

### Part B — surface speculative-decoding acceptance (fix 3)

**`BenchResult` gains two optional fields** (default `None`):

```python
draft_n: int | None = None           # speculative tokens proposed
draft_n_accepted: int | None = None  # of those, verified/kept
```

**Parsers stamp them when present:**

- `parse_llamacpp_timings` (toks:999): lift `timings.draft_n` /
  `timings.draft_n_accepted` (and read `timings.cache_n` for the guard below).
- `bench_from_sse` (toks:563) / the Unsloth streamed path: when an SSE event carries
  a `timings` block (Studio, or a llama.cpp impostor on the mlx port), lift the same
  fields; pure client-timed streams (real `mlx_lm`) leave them `None`.
- ollama / lmstudio: their responses expose no draft counters → stay `None`
  (`ACC%` renders `-`, exactly as a non-speculating backend should read).

**Display.** New `ACC%` column immediately right of `TOKENS/S`:

- `HEADER` (toks:1174): insert `"ACC%"` after `"TOKENS/S"`.
- `RIGHT_ALIGN` (toks:1176): add `"ACC%"`.
- `cached_acceptance(cache, rec)` helper mirroring `cached_tps`; returns
  `accepted / draft_n` when both present and `draft_n > 0`, else `None`.
- `build_rows` (toks:1179): render `f"{acc*100:.0f}%"` or `"-"`.

A populated `ACC%` is also the at-a-glance signal that a row's throughput is
spec-decode-assisted and therefore content-sensitive — the reader no longer has to
*know* that llama.cpp+MTP varies.

**cache_n inflation guard (couples B with fix 1).** If a timed sample reports
`timings.cache_n` greater than a few tokens (i.e. ≳ the nonce+prompt prefix that
should be unique), the nonce failed or an external cache replayed the prompt; the
number may be inflated. `run_benchmarks` prints a one-line warning naming the model
and `cache_n`, consistent with the existing `observed_backend` mismatch warning. It
records the number anyway (the warning, not suppression, is the contract).

### Part C — content spread across prompt kinds (fix 5, opt-in)

```python
CODE_PROMPT = ("Write a complete, thread-safe LRU cache in Python with O(1) get "
               "and put, per-entry TTL expiry, type hints, and docstrings. "
               "Output only the code.")
BENCH_PROMPTS = {"prose": DEFAULT_PROMPT, "code": CODE_PROMPT}
```

**Default is unchanged** (prose only) — keeps the common path lean and the table
narrow. `--bench-suite` (or `--bench-kinds prose,code`) measures each kind through
the Part A pipeline (nonce + N-median + acceptance).

**Cache (nested, back-compat).** `cache_key` stays `{provider}:{model}` — identity
in the value, not the key, per the prior spec's invariant. Suite results nest:

```json
"by_kind": {
  "prose": {"tokens_per_second": .., "draft_n": .., "draft_n_accepted": .., ..},
  "code":  {"tokens_per_second": .., "draft_n": .., "draft_n_accepted": .., ..}
}
```

The top-level `tokens_per_second` / `ACC%` continue to reflect the **prose** kind so
default rows and legacy readers are unaffected.

**Display.** When any displayed model has `by_kind.code`, two extra right-aligned
columns appear — `CODE/S` and `CODE%` — populated from `by_kind.code`; otherwise
they are omitted entirely (no empty churn for users who never run the suite). This
is the only rendering change and it is conditional.

## Cache schema (additive, fully back-compat)

`run_benchmarks` writes, in addition to today's fields:

| field | meaning |
|---|---|
| `tokens_per_second` | now the **median** of `samples` (same key, refined semantics) |
| `samples` | N actually measured |
| `tps_min`, `tps_max` | spread across samples |
| `draft_n`, `draft_n_accepted` | pooled speculative counters (`null` off-spec-decode) |
| `prompt` | base template (nonce stripped); `prompt_nonced: true` |
| `max_tokens` | now 256 |
| `by_kind` | present only under `--bench-suite` (Part C) |

All new keys are optional. `migrate_cache` (toks:1119) is unchanged: a legacy entry
reads as `samples` absent (≡ 1), `draft_*` absent (`ACC%` → `-`), no `by_kind`. A
re-bench overwrites in place under the same key, so a number that newly carries
acceptance/spread simply enriches the existing history.

## Non-goals

- **Forcing speculation off for a "raw" number.** The spec-decode rate *is* how
  Studio serves; measuring it is correct. The fix is making the sample
  representative and explained, not stripping the feature.
- **Seeded/reproducible prompts.** The nonce is intentionally random — independence,
  not repeatability, is the goal. Cross-machine numbers compare medians, not bytes.
- **Latency percentiles / per-token histograms.** Median + min/max is enough to flag
  a noisy or inflated row; distributions are out of scope.
- **Acceptance for ollama/lmstudio.** Their APIs don't expose draft counters; `-` is
  the honest rendering, not a gap to backfill.
- **Table grouping/sorting so a model's per-runtime rows sit adjacent** — still a
  separate rendering change (carried over from the prior spec's non-goals).

## Phasing

- **Phase 1 (default-on, no schema break):** Part A (nonce + N-median + 256) and
  Part B (`ACC%` + cache_n guard). Small, self-contained, converts the exact
  failure mode above into a visible, trustworthy signal. Ship first.
- **Phase 2 (opt-in):** Part C (`--bench-suite`, `by_kind`, conditional CODE
  columns). Lands once Phase 1 is in and the table impact is reviewed.

## Test plan (network-free, `unittest`)

Part A:
- `nonce_prompt` — two calls differ; the base prompt is preserved as the suffix; the
  unique bytes lead.
- `aggregate_samples` — median/min/max for odd and even N; single sample →
  median == that value, min == max == it; pooled acceptance == Σaccepted/Σdrafted;
  `draft_n` `None` when no sample carried draft stats.
- `run_benchmarks` runs N samples (mock `provider.benchmark` returning scripted
  `BenchResult`s, asserting it receives a *different* prompt each call) and writes
  `tokens_per_second` (median), `samples`, `tps_min`, `tps_max`; `--samples 1`
  reproduces today's single-shot value.
- guard test: `DEFAULT_MAX_TOKENS == 256`.

Part B:
- `parse_llamacpp_timings` populates `draft_n`/`draft_n_accepted` from a timings dict
  and leaves them `None` when the keys are absent.
- `bench_from_sse` lifts draft fields when a stream event carries a `timings` block;
  `None` for a pure client-timed stream.
- `cached_acceptance` returns 46/48 for a stored pair, `None` when `draft_n` is 0 or
  missing; `build_rows` renders `"96%"` and `"-"`; `HEADER`/`RIGHT_ALIGN` include
  `ACC%`.
- cache_n guard: `run_benchmarks` warns (capture stderr) when a sample's `cache_n`
  exceeds the prefix budget; no warning at `cache_n == 0`.

Part C:
- `BENCH_PROMPTS` has `prose` + `code`; `--bench-suite` selects both, default selects
  `prose` only.
- suite run writes `by_kind.{prose,code}`; top-level `tokens_per_second` equals the
  prose median; the conditional `CODE/S`/`CODE%` columns appear only when `by_kind`
  is present and are absent otherwise.
- a legacy entry (no `samples`/`draft_*`/`by_kind`) loads and migrates unchanged.
