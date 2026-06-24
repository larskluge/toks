# toks: Unsloth Studio provider + Bits/weight column — design

Date: 2026-06-24

## Summary

Two related additions to `toks`:

1. **`BPW` column** — effective bits/weight (`size_bytes × 8 ÷ param_count`),
   inserted after `PARAMS`. This surfaces the insight that two "4-bit" 31B models
   can differ sharply in real cost: a Q4_K_M GGUF lands at **4.78 bpw** while an
   MLX `nvfp4` build of the same model lands at **5.18 bpw**.
2. **`unsloth` provider** — support Unsloth Studio (an auth-gated, OpenAI-compatible
   llama.cpp server, default `http://127.0.0.1:8888`), benchmarked from its
   server-side `timings`, with size/quant/params/ctx recovered by parsing the
   model's cached GGUF header.

Supporting work: a read-only `.env` loader (`~/.config/toks/.env`) so the
`UNSLOTH_API_KEY` the user already stored there is picked up.

## Goals

- Add a `BPW` column populated for every row where on-disk size and a parameter
  count are both known (Ollama today; the new `unsloth` provider; `mlx` where the
  HF cache is readable).
- Add an `unsloth` provider: list, benchmark (accurate, server-measured), and
  enrich GGUF metadata from the local HF cache.
- Load `UNSLOTH_API_KEY` (and any other keys) from `~/.config/toks/.env`,
  **read-only** — never create or write that file.
- Keep the single-file, standard-library-only constraint.

## Non-goals / out of scope

- Re-pointing or fixing the existing `mlx` provider, whose default `MLX_URL`
  (`:8080`) currently happens to target a llama.cpp server on this machine. Noted
  but untouched.
- Writing/creating the `.env` file. The user created it; `toks` only reads it.
- Any column reorder or rename. The change is purely **additive** (one new column).
- Reading GGUF metadata over the network. Enrichment is HF-cache-local only, exactly
  like the `mlx` provider.

## Current state (relevant pieces)

- `ModelRecord` (toks:44) holds a `params` **display string** (e.g. `"8B"`,
  `"~31B"`, `"31B MoE 8/256"`) but **no numeric** count. BPW needs the number.
- `ollama_params` (toks:294) computes a numeric `total` then formats it, returning
  only the string. `mlx_enrich` (toks:423) likewise computes `total` then discards it.
- `human_params` (toks:257) formats a numeric total into the display string.
- `quant_bits_per_weight` (toks:237) maps a quant label to an approximate stored
  width — used only to *estimate* params from size; not a per-row measured value.
- `http_sse` (toks:723) streams an SSE body but **takes no headers** (so it can't
  send an auth token yet).
- Providers are plain classes selected in `select_providers` (toks:870); cache keys
  are namespaced by provider prefix in `_PROVIDER_PREFIXES` (toks:546).
- `HEADER`/`RIGHT_ALIGN`/`build_rows`/`table` (toks:615–656) drive rendering.

## Design

### 1. Data model

Add two fields to `ModelRecord`:

```python
param_count: int | None = None     # numeric total params (real or estimated)
param_estimated: bool = False      # True when param_count was derived from size
```

`params` (the display string) stays as-is. `param_count` is the new numeric source
for BPW. `param_estimated` mirrors the existing `~` convention so BPW can show `~`
on rows whose param count was assumed rather than read.

### 2. BPW column

- **Compute:** `effective_bpw(rec)` returns `rec.size_bytes * 8 / rec.param_count`
  when both are present and `> 0`, else `None`.
- **Render:** `human_bpw(rec)` → `"-"` when `None`, else `f"{bpw:.2f}"`, prefixed
  with `~` when `rec.param_estimated` is true (e.g. `4.78`, `5.18`, `~4.50`).
- **Placement:** insert `"BPW"` into `HEADER` immediately after `"PARAMS"`, add it
  to `RIGHT_ALIGN`, and emit the cell in `build_rows` in the same position. No other
  column moves.

**Semantics note.** BPW is *measured* when `param_count` is real (Ollama's
`general.parameter_count`; the `unsloth` GGUF tensor-sum). When `param_count` was
estimated from size via a nominal width (the existing `~` paths), BPW is necessarily
≈ that nominal width — the leading `~` flags this, consistent with `PARAMS`.

**Populating `param_count`:**
- `ollama_params` is refactored to expose the numeric total. Cleanest split: a new
  `ollama_param_count(model, info) -> (total, estimated, expert_count, expert_used)`
  that holds the existing real-vs-estimated logic (toks:294–326), with `ollama_params`
  /`ollama_parse_models` calling it to set both `params` (via `human_params`) and the
  new `param_count`/`param_estimated`. Behaviour of the display string is unchanged.
- `mlx_enrich` already computes `total` (toks:471); it now also sets
  `rec.param_count = total` and `rec.param_estimated = True` (its counts are always
  size-derived, hence the existing `~`).
- `unsloth` enrichment sets `rec.param_count` from the GGUF tensor-sum with
  `param_estimated = False` (exact).

### 3. Unsloth Studio provider

**Server shape (verified):** Unsloth Studio is a llama.cpp build
(`system_fingerprint: b9773-…`) speaking the OpenAI API behind a bearer token.
- `GET /v1/models` → ids only (meta empty); the id is an HF repo id, e.g.
  `unsloth/gemma-4-31B-it-GGUF`.
- `POST /v1/completions` (stream, `stream_options.include_usage`) → final chunk
  carries a llama.cpp `timings` object with `predicted_per_second` and `prompt_ms`.

**`UnslothProvider` (name `"unsloth"`):**
- **URL:** `normalize_host(UNSLOTH_URL, "http://127.0.0.1:8888")`.
- **Auth:** `unsloth_headers()` → `{"Authorization": f"Bearer {UNSLOTH_API_KEY}"}`
  when the key is set, else `{}`. (Naming follows the user's existing env var.)
- **`list_models`:** `GET /v1/models` (with auth) → `unsloth_parse_models` builds
  records (`provider="unsloth"`, `benchmarkable=True`). When the host is local
  (`is_local_host`), enrich each record from the HF cache (below).
- **`benchmark`:** 1-token warmup (harmless if already loaded), then stream
  `/v1/completions` with `include_usage` and auth headers. Prefer server-side
  timings: `parse_llamacpp_timings(final_usage_chunk)` →
  `BenchResult(timings["predicted_per_second"], timings["prompt_ms"]/1000)`. If no
  `timings` block is present, fall back to the existing client-side
  `bench_from_sse`. This requires threading the SSE stream through the timings
  parser; the benchmark reads the raw SSE lines once and extracts both.

**HTTP change:** add an optional `headers=None` parameter to `http_sse` (toks:723)
so the auth token can be sent. Backward compatible (existing `mlx` call passes none).

**GGUF cache enrichment (`unsloth_enrich(records, hub_root)`):**
- Locate the snapshot with the existing `_mlx_snapshot(hub_root, model_id)`
  (handles HF repo ids and absolute-path ids). Absent repo → leave row as `-`.
- **Pick the main weights GGUF.** A snapshot may hold several `.gguf` files
  (verified: main `…-Q4_K_M.gguf` 18.3 GB plus `mmproj-*.gguf` vision projectors
  and an `mtp-*.gguf` draft head). Heuristic: among `*.gguf`, drop names containing
  `mmproj` or `mtp`, then take the **largest** remaining. That file's size is
  `size_bytes` (≈18.3 GB), not the snapshot sum.
- **Parse its GGUF header** (`read_gguf_meta(path)`, below) for: exact
  `param_count`, `file_type` → quant label, context length, MoE flags.
- Set `rec.fmt = "gguf"`, `rec.size_bytes`, `rec.quant`, `rec.param_count`
  (`param_estimated=False`), `rec.params = human_params(...)`, `rec.ctx_max`, and
  `rec.modified_at` from the ref mtime (as `mlx_enrich` does). On any parse failure,
  leave the record untouched — never raise (best-effort, like `mlx_enrich`).

### 4. GGUF header reader (`read_gguf_meta`)

A small, self-contained, stdlib `struct` parser of the GGUF v2/v3 header (verified
against the cached file: 833 tensors → 30,697,345,596 params, file_type 15 → Q4_K_M):

1. Read magic `GGUF`, version, `tensor_count`, `kv_count` (little-endian).
2. Walk `kv_count` key/value pairs (full GGUF type table incl. arrays). Capture
   `general.file_type`, `*.expert_count` / `*.expert_used_count`, and
   `<arch>.context_length`. Other keys are read-and-discarded (needed only to reach
   the tensor section).
3. Walk `tensor_count` tensor-info entries (`name`, `n_dims`, `dims[]`, `type`,
   `offset`); `param_count = Σ Π(dims)`. Tensor data itself is never read.

Returns a dict `{param_count, file_type, quant, ctx, expert_count, expert_used}`;
returns `None` / partial on malformed input. File-type→quant uses the standard
llama.cpp enum (e.g. `15 → "Q4_K_M"`); unknown ids fall back to a generic label.

### 5. `.env` loader (read-only)

`load_dotenv()`:
- Path: `${XDG_CONFIG_HOME:-~/.config}/toks/.env`.
- If absent/unreadable: no-op.
- Parse `KEY=VALUE` lines: skip blanks and `#` comments, tolerate a leading
  `export`, strip surrounding single/double quotes from the value.
- Set into `os.environ` **only if the key is not already present** — a real
  environment variable always wins. **Never writes** the file.
- Called once at the top of `main()`, before any `*_url()` / header lookups.

No repo `.env` is created, so `.gitignore` is unchanged and no secret enters the
repo. (For defense in depth we may still add `.env` to `.gitignore`; optional.)

### 6. Wiring

- `select_providers` (toks:870): add `"unsloth": UnslothProvider`.
- `parse_args` `--provider` choices (toks:993): add `"unsloth"`.
- `_PROVIDER_PREFIXES` (toks:546): add `"unsloth:"` for cache namespacing.
- `tag_cell` (toks:531): `fmt="gguf"` already renders `gguf` (+ `moe`); no change
  needed beyond setting `rec.fmt = "gguf"` during enrichment.

## Example output (illustrative)

```
PROVIDER  NAME                         SIZE  TAG    PARAMS   BPW   CTX   TTFT  TOKENS/S  MODIFIED
unsloth   unsloth/gemma-4-31B-it-GGUF  17.1 GB  gguf     31B  4.78  256k  0.54      39.6  2 days ago
ollama    gemma4:31b-mlx               18.8 GB  mlx      31B  5.18  256k  0.08      24.0  1 day ago
```

Same model, same `31B`, same `256k` — the new `BPW` column is what distinguishes
the Q4_K_M GGUF (4.78) from the MLX `nvfp4` build (5.18). (`SIZE` is toks' existing
binary/GiB rendering, so 18.3 GB on disk shows as `17.1 GB`; `BPW` is computed from
exact raw bytes, hence 4.78.)

## Error handling

- All enrichment and GGUF parsing is best-effort and never raises (matches
  `mlx_enrich`): malformed headers, missing files, multi-GGUF snapshots, and remote
  (non-local) hosts all degrade gracefully to `-`.
- Benchmarking against a remote `unsloth` host still works (server-side `timings`),
  even though enrichment columns will be `-`.
- A missing/invalid `UNSLOTH_API_KEY` yields the existing `401` RuntimeError path
  (`http_json`), and the provider is skipped with a one-line stderr note via
  `gather_records` — `toks` still lists other backends.

## Testing (network-free, `test_toks.py`)

- `read_gguf_meta`: synthesize a minimal valid GGUF byte blob in-memory (a couple
  KVs incl. `general.file_type`, two tensors with known dims) and assert exact
  `param_count`, quant mapping, ctx, MoE detection; assert graceful `None` on
  truncated/garbage input.
- Main-GGUF selection: given a fake file set incl. `mmproj-*`/`mtp-*`, assert the
  largest non-projector file is chosen.
- `parse_llamacpp_timings`: maps a sample `timings` dict to the right
  `BenchResult` (tps + ttft); `None` on missing/zero fields.
- `effective_bpw` / `human_bpw`: exact value, 2-dp formatting, `~` prefix when
  `param_estimated`, `-` when size or count missing.
- `unsloth_parse_models`: ids-only payload → records with provider/benchmarkable.
- `load_dotenv`: temp file with comments/quotes/`export` → correct `os.environ`
  population; pre-set keys are **not** overridden; absent file is a no-op.
- `ollama_param_count` refactor: existing Ollama param tests still pass; add an
  assertion that `param_count`/`param_estimated` are set correctly (real vs
  estimated-from-size).
- Rendering: `HEADER` contains `BPW` after `PARAMS`; `build_rows` emits the cell;
  update any existing test that pins header length or row contents.

## Files changed

- `toks` — data model, BPW helpers + column, `UnslothProvider`, `read_gguf_meta`,
  `unsloth_enrich`, `unsloth_parse_models`, `parse_llamacpp_timings`, `http_sse`
  headers param, `load_dotenv`, wiring.
- `test_toks.py` — new tests above; adjust existing for the BPW column / refactor.
- `README.md` — `unsloth` in provider list/choices; config table rows for
  `UNSLOTH_URL` / `UNSLOTH_API_KEY` and the `~/.config/toks/.env` loader; BPW column
  in the example table; a short "Unsloth Studio prerequisites" note.
- `docs/specs/2026-06-24-unsloth-studio-provider-and-bpw-design.md` — this file.
