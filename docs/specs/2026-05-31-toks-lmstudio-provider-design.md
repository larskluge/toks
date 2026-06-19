# toks — multi-backend local-model lister (Ollama + LM Studio)

Date: 2026-05-31
Status: Draft for review
Target file: `ollama-list-with-tps` → renamed to `toks`
(currently invoked as `ols`; see §12 Renaming)

## 1. Goal

Rename the tool to **`toks`** (it is no longer Ollama-specific) and add **LM
Studio** as a second model backend alongside Ollama, in a **single unified, merged
listing** — one table with a `PROVIDER` column, sorted by tokens/sec across *both*
backends. This makes `toks` a cross-backend speed comparison tool (e.g. GGUF on
Ollama vs MLX on LM Studio on the same machine).

Non-goals: model management (load/unload/download), chat/serving, or non-local
providers. We only **list** and **benchmark**.

## 2. Constraints (decided)

- **Language: Python 3, standard library only** — keep the existing `#!/usr/bin/env
  python3` script and refactor it in place (no rewrite, no third-party deps). Allowed
  modules stay limited to what's already used: `argparse`, `json`, `os`, `sys`,
  `time`, `urllib`, `concurrent.futures`, `dataclasses`, `datetime`, `pathlib`.
- **Unified merged list** with a `PROVIDER` column; sort by TPS across both backends.
- **Host must be configurable.** (Originally assumed a remote host; in
  practice the live server runs locally on `127.0.0.1:1234`. Either works — the
  host is just an env var — and server-side stats keep remote benching accurate.)
- **Pure HTTP**, Python stdlib only, matching the current tool's zero-dependency
  promise. (`lms` CLI is present but deliberately unused, to stay HTTP-only.)
- **LM Studio native v0 API** is the baseline: `GET /api/v0/models` for listing,
  `POST /api/v0/completions` for benchmarking.
- **Backward compatible**: with no LM Studio reachable, `toks` behaves exactly as it
  does today (Ollama-only), including preserving the existing benchmark cache.

## 3. What LM Studio exposes (research summary)

Default endpoint `http://<host>:1234`, no key for local (optional Bearer via
`LM_API_TOKEN` / `Authorization: Bearer`). Two API surfaces:

- **OpenAI-compat** (`/v1/...`): `GET /v1/models` returns only `id`/`object`; chat
  returns `usage` but **no speed stats**. Not useful — ignored.
- **Native** (`/api/v0/...`, plus newer `/api/v1/...` in 0.4.0+): rich metadata +
  server-measured throughput stats. **This is what we use.**

### 3.1 Listing — `GET /api/v0/models`

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2-vl-7b-instruct",
      "object": "model",
      "type": "vlm",                  // llm | vlm | embeddings
      "publisher": "mlx-community",
      "arch": "qwen2_vl",
      "compatibility_type": "mlx",    // mlx | gguf  -> FMT column
      "quantization": "4bit",         // -> QUANT column
      "state": "not-loaded",          // loaded | not-loaded | loading
      "max_context_length": 32768,
      "loaded_context_length": 32768  // present only when loaded
    }
  ]
}
```

**Gap:** the v0 listing has **no on-disk size, no parameter count, no modified
date**. Those are recovered *best-effort* from the richer `GET /api/v1/models`
(0.4.0+). **Live-captured shape (differs from the docs):** the v1 payload nests
under **`models`** (not `data`) and keys each entry by **`key`** (not `id`):

```json
{
  "object": "list",
  "models": [
    {
      "key": "gpt-oss-120b",
      "architecture": "gpt_oss",
      "quantization": { "name": "MXFP4", "bits_per_weight": 4 },
      "size_bytes": 62358097894,
      "params_string": "120B",
      "format": "mlx",
      "max_context_length": 131072
    }
  ]
}
```

The enricher matches v1 `key` → v0 `id` and backfills `size_bytes` /
`params_string` (tolerating the `data`/`id` shape too, in case a version differs).
If v1 is absent or omits a model, `SIZE`/`PARAMS` show `-`. Degrades silently —
never an error. Embeddings models carry no `params_string`, so PARAMS stays `-`.

### 3.2 Benchmarking — `POST /api/v0/completions`

We mirror the current Ollama bench (raw completion via `/api/generate`) with the
analogous raw endpoint. Response includes **server-measured** stats:

```json
{
  "usage": { "prompt_tokens": 12, "completion_tokens": 128, "total_tokens": 140 },
  "stats": {
    "tokens_per_second": 51.43,
    "time_to_first_token": 0.111,
    "generation_time": 0.954,
    "stop_reason": "eosFound"
  },
  "model_info": { "arch": "...", "quant": "...", "format": "...", "context_length": 0 },
  "runtime":    { "name": "...", "version": "...", "supported_formats": ["..."] }
}
```

vs Ollama, where `toks` computes `tps = eval_count / eval_duration`. LM Studio hands
us `tokens_per_second` directly **plus** free `time_to_first_token`.

**Remote-host note:** `tokens_per_second` and `time_to_first_token` are measured
*server-side* by LM Studio, so network latency to a remote host does **not**
distort them — only wall-clock would be affected, which we don't use. Benching a
remote host (over any network) is therefore fair. (The live server in
this case runs locally on `127.0.0.1:1234`; both work.)

## 4. Architecture

Keep a **single file, stdlib-only**. Introduce a thin **provider abstraction** so
Ollama and LM Studio are interchangeable, and the listing/cache/render layers work
on a normalized record.

### 4.1 Normalized model record

```python
@dataclass
class ModelRecord:
    provider: str          # "ollama" | "lmstudio"
    name: str              # display id (Ollama tag, or LM Studio model id)
    fmt: str               # gguf | safetensors | mlx | ""
    params: str            # "8.2B" | "-"
    moe: bool
    quant: str             # "Q4_K_M" | "4bit" | "-"
    size_bytes: float|None # on-disk size, None if unknown (LM Studio v0)
    modified_at: str|None  # ISO ts, None for LM Studio
    ctx_max: int|None      # max context length
    ctx_loaded: int|None   # loaded context (LM Studio loaded models)
    benchmarkable: bool
    raw: dict              # original payload, for provider-specific needs
```

### 4.2 Provider interface

```python
class Provider:                      # base / informal protocol
    name: str
    def available(self) -> bool: ...                 # cheap reachability probe
    def list_models(self) -> list[ModelRecord]: ...  # + best-effort enrichment
    def benchmark(self, rec, prompt, max_tokens) -> BenchResult | None: ...

@dataclass
class BenchResult:
    tokens_per_second: float
    time_to_first_token: float | None
```

- **`OllamaProvider`** — refactor existing code behind this interface,
  behavior-preserving:
  - `available()`: `GET {OLLAMA_HOST}/api/tags` succeeds.
  - `list_models()`: `/api/tags` → records; parallel `/api/show` enrichment
    (existing `ThreadPoolExecutor(8)`) for params/MoE/arch/ctx.
  - `benchmark()`: `POST /api/generate`, `tps = eval_count/eval_duration`; also
    capture `time_to_first_token` from `prompt_eval_duration` (+`load_duration`
    when present) for the new TTFT column.
  - `benchmarkable`: `fmt in {gguf, safetensors}` (unchanged).

- **`LMStudioProvider`** (new):
  - Host: `LMSTUDIO_HOST` env, default `http://localhost:1234`; same normalization
    as `ollama_host()` (prepend `http://` if no scheme, strip trailing `/`).
    Optional `Authorization: Bearer` from `LMSTUDIO_API_KEY` or `LM_API_TOKEN`.
  - `available()`: `GET /api/v0/models` succeeds (short timeout).
  - `list_models()`: `GET /api/v0/models` → records (fmt=`compatibility_type`,
    quant, ctx_max/loaded, type). Then **best-effort** `GET /api/v1/models`; if it
    returns `size_bytes`/`params_string`, backfill `size_bytes`/`params` by id.
    Failures here are swallowed (columns stay `-`).
  - `benchmark()`: **warmup** call (`max_tokens=1`) to trigger JIT load, then the
    measured `POST /api/v0/completions` (`{model, prompt, max_tokens, stream:false,
    temperature:0}`); read `stats.tokens_per_second` and `stats.time_to_first_token`.
    If `stats` is absent (older server), return `None` (no unreliable wall-clock fallback).
  - `benchmarkable`: `type in {llm, vlm}` (skip `embeddings`).

### 4.3 Shared HTTP helper

Generalize the current `api_json(path, payload, timeout)` (which hardcodes
`ollama_host()`) into `http_json(url, payload=None, headers=None, timeout=...)`
taking a full URL + optional headers. Each provider builds its own URLs/headers.
Keep `urllib` + `json` only.

### 4.4 Orchestration (`main`)

- New flag `--provider {ollama,lmstudio,all}` (default `all`).
- Build the active provider list; for `all`, include each provider that
  `available()` returns true for. Print a one-line **stderr** note for a
  configured-but-unreachable provider; **error only if none** are reachable.
- Gather `list_models()` from each active provider (per-provider failures are
  isolated and noted, never abort the others), merge, sort by `tps_sort_key`
  across all rows.
- `--bench` targeting (`--all` / `--missing` / `--model`) applies across active
  providers; `--model NAME` matches within any provider (benches all matches).
- Update the argparse `description` from "List Ollama models with a cached
  tokens/sec column." to "List local models (Ollama + LM Studio) with a cached
  tokens/sec column."

## 5. Output

The tool **today** prints these columns: `NAME  SIZE  TAG  PARAMS  TOKENS/S
MODIFIED`, where `TAG` is a combined badge (e.g. `gguf`, `mlx`, `mlx moe`) built by
`model_tag()` from format + MoE detection. There is **no** `--json` and **no**
`--no-color` flag today, and this change does not add them.

Changes:
- Add `PROVIDER` as the leading column.
- `TAG` already emits `mlx`/`gguf`/`moe`; LM Studio maps `compatibility_type` →
  `mlx`/`gguf`, plus `moe` when arch indicates it.
- Add `TTFT` and `CTX` columns (both backends can supply them; `-` when unknown).
  Ollama: `CTX` from `<arch>.context_length`, `TTFT` from `prompt_eval_duration`
  (+`load_duration`). LM Studio: `CTX` from `loaded`/`max_context_length`, `TTFT`
  from `stats.time_to_first_token`.
- `SIZE`/`PARAMS`/`MODIFIED` show `-` for LM Studio v0 rows unless the best-effort
  v1 enrichment fills `SIZE`/`PARAMS`.

```
PROVIDER  NAME          SIZE    TAG       PARAMS  CTX     TTFT  TOKENS/S  MODIFIED
lmstudio  qwen3-8b-mlx  -       mlx       8.2B    32768   0.11     188.0  -
ollama    qwen3:8b      4.7 GB  gguf      8.2B    40960   0.21     142.0  3 days ago
ollama    llama3.1:8b   4.7 GB  gguf      8.0B    8192    0.25     121.0  8 days ago
lmstudio  gemma-3-12b   -       gguf      12.0B   8192    0.30      77.0  -
```

- `-` for any unavailable cell. `right_align` set extends to include `CTX`, `TTFT`.

## 6. Cache

The cache today is a **flat dict** `{key: entry}` with `key = digest or name` and
**no version field** (`load_cache()` just returns the dict). The key collides across
providers once LM Studio is added. Changes:

- Key becomes `f"{provider}:{digest-or-id}:{quant}"`.
- **Migrate on load, no data loss:** any existing key that contains no `:` is a
  legacy Ollama entry — re-key it to `ollama:<old-key>` and stamp
  `entry["provider"]="ollama"`. Write back in the new format. (LM Studio ids can
  contain no `:`, but legacy entries predate any LM Studio rows, so this is safe;
  the migration runs once.)
- Entry schema gains `provider` and `time_to_first_token`; keeps `model`,
  `tokens_per_second`, `benchmarked_at`, `prompt`, `max_tokens`.

## 7. Configuration & prerequisites

| Setting | Env / flag | Default | Notes |
|---|---|---|---|
| Ollama host | `OLLAMA_HOST` | `127.0.0.1:11434` | unchanged |
| LM Studio host | `LMSTUDIO_HOST` | `localhost:1234` | set to a host name/IP, e.g. `<machine>.<domain>:1234` or `192.168.x.y:1234` |
| LM Studio token | `LMSTUDIO_API_KEY` / `LM_API_TOKEN` | none | optional `Authorization: Bearer` |
| Provider select | `--provider` | `all` | `ollama` \| `lmstudio` \| `all` |

**Remote prerequisite (document in README):** on the LM Studio host, the server
must be started and set to **"Serve on Local Network"** (bind `0.0.0.0`, not just
`localhost`) so it is reachable over the network. Port 1234 must be allowed by the
host firewall / reachable from the client.

## 8. Error handling

- Per-provider isolation: listing/benching failures in one provider never abort the
  other; emit a concise stderr line and continue.
- LM Studio specifics:
  - connection refused / timeout → "LM Studio unreachable at <host> (is the server
    running and serving on the local network?)".
  - `401` → hint to set `LMSTUDIO_API_KEY`.
  - `/api/v1/models` errors during enrichment → swallowed silently.
- Timeouts: keep `BENCH_TIMEOUT=600s` (JIT load of large models is slow); use a
  modest remote listing timeout (~10s) instead of the current local 5s.

## 9. Testing

The tool has no tests today. Add a minimal, network-free `unittest` suite (stdlib,
runnable via `python3 -m unittest`) plus a manual checklist.

Unit tests (with **anonymized** captured fixtures):
- `LMStudioProvider` parse: `/api/v0/models` JSON → `ModelRecord` (fmt/quant/ctx/type).
- v1 enrichment backfill: `size_bytes`/`params_string` merged by id; absent v1 → `-`.
- Bench parse: `stats` JSON → `BenchResult` (tps + ttft); missing `stats` → `None`.
- Cache: new key format; **v1→v2 migration** preserves Ollama entries.
- Merge/sort: mixed-provider records sort by TPS, unbenched last.
- Ollama provider parse unchanged (regression guard on existing behavior).

Manual verification (against the live remote host):
- `LMSTUDIO_HOST=http://<host>:1234 toks` lists LM Studio + Ollama merged.
- `toks --provider lmstudio --bench --all` populates TPS/TTFT from stats.
- With LM Studio host unreachable, `toks` still lists Ollama and prints one note.

## 10. Backward compatibility

- No LM Studio reachable → identical to today's output (Ollama-only), existing cache
  preserved via migration.
- All current flags keep their meaning; `--provider` and the new columns are additive.
- Single file, no third-party dependencies — README's promise intact.

## 11. Open risks

- Whether `/api/v1/models` actually returns `size_bytes`/`params_string` varies by
  LM Studio version; design treats it as best-effort so SIZE/PARAMS gracefully blank.
- `--model NAME` could match the same name in both providers; we bench all matches
  (acceptable; documented).

## 12. Renaming to `toks`

The tool is renamed from `ollama-list-with-tps` / `ols` to **`toks`** (it now spans
backends). Scope of the rename:

- **Script:** rename `ollama-list-with-tps` → `toks`
  (the dev copy). The installed copy/symlink under `~/bin` (`ols` →
  `ollama-list-with-tps`) is re-pointed to `toks`; keep an `ols` symlink/alias as a
  back-compat shim so muscle memory and the existing shell alias keep working.
- **Cache dir:** currently `~/.cache/ollama-list-with-tps/tokens-per-second.json`.
  Move to `~/.cache/toks/tokens-per-second.json`. On first run, if the new path is
  absent but the old one exists, **migrate** (copy/rename) it so cached benchmarks
  survive — combine with the §6 key migration in one pass.
- **Code:** `cache_path()` uses the new dir name; argparse `description` updated
  (§4.4); any "Ollama"-specific user-facing strings generalized where they now
  cover both backends.
- **Out of scope / manual:** the user's interactive shell alias `ols` (in their
  shell rc) — note it for the user; the spec does not edit shell rc files.
