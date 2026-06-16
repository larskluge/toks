# toks

List local LLMs across backends — **Ollama**, **LM Studio**, and **mlx-lm**
(`mlx_lm.server`) — in one table, ranked by cached tokens/sec. Single file,
Python 3 standard library only.

```
$ toks
PROVIDER  NAME             SIZE     TAG       PARAMS  CTX     TTFT  TOKENS/S  MODIFIED
mlx       org/chat-3b      1.7 GB   mlx       ~3B     131072  0.09     204.2  19 minutes ago
lmstudio  qwen3-8b-mlx     -        mlx       8B      32768   0.11     188.0  -
ollama    qwen3.6:35b-mlx  20.4 GB  mlx moe   35B     262144  0.18     129.5  1 day ago
ollama    gpt-oss:120b     60.9 GB  gguf moe  117B…   131072  0.402     79.6  2 hours ago
```

## Usage

```
toks                            # list models from all reachable backends
toks --provider ollama          # one backend only (ollama | lmstudio | mlx | all)
toks --bench                    # benchmark models with no cached value (= --bench missing)
toks --bench all                # benchmark every model, cache the result
toks --bench qwen3.6:27b-mlx    # benchmark the named model(s)
```

Benchmarks are cached at `~/.cache/toks/tokens-per-second.json`, so plain `toks`
is instant and the TOKENS/S / TTFT columns persist between runs. Rows are sorted
fastest-first; un-benchmarked models sort last.

With `--bench`, the printed table is restricted to the models that were
benchmarked (the missing set, `all`, or the named models). Run plain `toks` to
see the full listing.

For Ollama, throughput is computed from the `/api/generate` eval counters. For LM
Studio, `tokens_per_second` and `time_to_first_token` are read straight from the
server's own `stats` — so benchmarking a **remote** LM Studio (e.g. on another host)
stays accurate, since those numbers are measured server-side. mlx-lm reports no
server-side stats, so `toks` times the SSE stream client-side (TTFT = first chunk,
throughput over the first→last chunk interval) — accurate for the localhost default;
over a network, latency leaks into the numbers.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LMSTUDIO_URL` | `http://localhost:1234` | LM Studio endpoint; set to a URL for a remote host, e.g. `http://box.example.net:1234` |
| `LMSTUDIO_API_KEY` / `LM_API_TOKEN` | — | optional `Authorization: Bearer` token |
| `MLX_URL` | `http://localhost:8080` | mlx_lm.server endpoint |
| `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | HF cache used for mlx metadata enrichment |
| `XDG_CACHE_HOME` | `~/.cache` | cache location root |

A bare host (no scheme) is accepted — `http://` is assumed.

## LM Studio prerequisites

1. Start the server: `lms server start` (or toggle it on in the app's Developer tab).
2. To reach it from another machine, enable **"Serve on Local Network"** so it
   binds `0.0.0.0` rather than just `localhost`, and make sure port `1234` is
   reachable (firewall / network).
3. LM Studio JIT-loads a model on first request; `toks` sends a 1-token warmup
   before each benchmark so model-load time doesn't pollute the measured stats.

The native v0 API (`/api/v0/models`, `/api/v0/completions`) is used for listing and
benchmarking. The v0 listing omits on-disk size and parameter count; `toks` makes a
best-effort backfill from the richer `/api/v1/models` (LM Studio 0.4.0+) and shows
`-` where that information isn't available.

## mlx-lm prerequisites

1. Start the server: `mlx_lm.server` (default port `8080`; ships with the
   [mlx-lm](https://github.com/ml-explore/mlx-lm) package).
2. Its `/v1/models` listing carries only model ids, so `toks` enriches rows from
   the local **HF cache** (size, quant, context, MoE shape, estimated `~` params) —
   this works only when `toks` runs on the same machine as the server; for a remote
   `MLX_URL` those columns show `-`.
3. The server's cache scan can list non-LLM repos (e.g. diffusers image models);
   `toks` detects these locally and refuses to benchmark them.
4. Models JIT-load on first request; the usual 1-token warmup absorbs the load time.

## Behaviour with one backend down

`toks` (provider `all`) lists whatever is reachable and prints a one-line note to
stderr for any configured-but-unreachable backend. It only errors if **no** backend
is reachable. So with LM Studio and mlx-lm off, it behaves exactly like the
original Ollama-only tool.

## Tests

```
python3 -m unittest test_toks   # network-free unit tests
```

## Design

See `docs/superpowers/specs/2026-05-31-toks-lmstudio-provider-design.md` and
`docs/superpowers/specs/2026-06-06-mlx-provider-design.md`.
