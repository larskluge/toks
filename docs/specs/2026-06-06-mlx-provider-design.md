# toks — mlx provider (mlx_lm.server)

Date: 2026-06-06
Status: Approved
Target file: `toks` (single-file, stdlib-only — constraint unchanged)

## 1. Goal

Add **mlx_lm.server** (the HTTP server shipped with Apple's
[mlx-lm](https://github.com/ml-explore/mlx-lm)) as a third provider alongside
Ollama and LM Studio. mlx-lm is the upstream engine behind LM Studio's MLX
runtime and typically gains new model support first, so benching it raw against
the wrappers is exactly toks's purpose.

Non-goals: managing the server, auth (mlx_lm.server has none), model
download/load management.

## 2. What mlx_lm.server exposes (probed live, v0.31.3)

- `GET /v1/models` — scans the HF cache for repos whose file *names* include
  `config.json`, `model.safetensors.index.json`, `tokenizer_config.json`
  (any directory level, so diffusers repos like image models false-positive).
  Returns only `{id, object, created}` — **no size/quant/ctx/params/type**.
  Also appends the `--model` CLI arg as an absolute-path id when it is a local
  path.
- `POST /v1/completions` — OpenAI shape; returns `usage` but **no server-side
  speed stats** (unlike LM Studio). Supports `stream: true` and
  `stream_options: {"include_usage": true}` (final chunk carries `usage`).
- Models JIT-load on demand per request. `GET /health` exists.

## 3. Design

### 3.1 Provider

`MlxProvider` (`name = "mlx"`), same informal interface as the other two.
Host from `MLX_URL`, default `http://localhost:8080`, via `normalize_host`.
No auth headers. `--provider` gains `mlx`; default `all` includes it.
Cache key `mlx:<model-id>`; `"mlx:"` added to `_PROVIDER_PREFIXES`.

### 3.2 Listing + local HF-cache enrichment

`list_models()`: `GET /v1/models` → records with `fmt="mlx"`,
`benchmarkable=True`, everything else unknown. Then, **only when the host is
local** (`localhost` / `127.0.0.1` / `::1`), enrich from the HF cache (root:
`HF_HUB_CACHE`, else `HF_HOME/hub`, else `~/.cache/huggingface/hub`):

- Resolve `org/name` → `models--org--name`, follow `refs/main` → snapshot dir.
  Absolute-path ids are used directly as the snapshot dir.
- `SIZE`: sum of symlink-resolved file sizes in the snapshot.
- `config.json` (top level): `quantization.bits` → quant `"4bit"`/`"8bit"`
  (bpw = bits + 0.5; no quantization key → 16 bpw); `max_position_embeddings`
  → CTX; any top-level `*expert*` int > 1 → MoE, with `num_experts` /
  `num_experts_per_tok` feeding `human_params`'s `MoE n/m` suffix.
- `PARAMS`: estimated from size ÷ bpw, `~`-prefixed (same convention as the
  Ollama estimate path).
- `benchmarkable`: demoted to `False` when the top-level `config.json` is
  missing or has no `model_type` — kills the diffusers/image-model false
  positive. (Remote hosts keep the optimistic `True`; a failed bench warns.)
- `MODIFIED`: mtime of `refs/main` (download time), ISO 8601.
- All enrichment is best-effort and silent; failures leave columns `-`.

### 3.3 Benchmark (client-side streaming timing)

No server stats, so time client-side over SSE — accurate for the localhost
default, and TTFT/generation are cleanly separated:

1. Warmup completion (`max_tokens=1`) absorbs the JIT model load (mirrors the
   LM Studio provider).
2. `POST /v1/completions` with `stream: true`,
   `stream_options: {"include_usage": true}`, `temperature: 0`.
3. A small stdlib SSE reader times each `data:` content chunk:
   - `TTFT` = first content chunk − request start.
   - `TOKENS/S` = `(completion_tokens − 1) / (last − first)`, with
     `completion_tokens` from the final `usage` chunk; falls back to counting
     content chunks when `usage` is absent. Returns `None` if nothing usable.

### 3.4 Errors

Unreachable server → existing per-provider skip note. Bench failure → existing
warning path. Enrichment failures silent.

## 4. Testing

Network-free `unittest` additions, anonymized fixtures:

- `/v1/models` payload → records (defaults, empty payload).
- Enrichment against a fake hub layout in a tmpdir: size (symlink resolved),
  quant/ctx/MoE/params from config.json, missing-config → not benchmarkable,
  unknown id untouched, modified_at set.
- SSE bench parse with a fake clock: tps+ttft from usage, chunk-count
  fallback, no-content → None, malformed lines skipped.
- `select_providers("mlx")` / `"all"`; `--provider mlx` accepted; cache-key
  prefix.

Manual: live server on :8080 — listing merges three providers;
`--bench` populates TPS/TTFT; image-model repo listed but not benchable.

## 5. README

Document `MLX_URL`, the local-only enrichment caveat, and the
`mlx_lm.server` prerequisite.
