# toks

List local LLMs across backends — **Ollama** and **LM Studio** — in one table,
ranked by cached tokens/sec. Single file, Python 3 standard library only.

```
$ toks
PROVIDER  NAME             SIZE     TAG       PARAMS  CTX     TTFT  TOKENS/S  MODIFIED
lmstudio  qwen3-8b-mlx     -        mlx       8B      32768   0.11     188.0  -
ollama    qwen3.6:35b-mlx  20.4 GB  mlx moe   35B     262144  0.18     129.5  1 day ago
ollama    gpt-oss:120b     60.9 GB  gguf moe  117B…   131072  0.402     79.6  2 hours ago
```

## Usage

```
toks                            # list models from all reachable backends
toks --provider ollama          # one backend only (ollama | lmstudio | all)
toks --bench                    # benchmark models with no cached value (= --bench missing)
toks --bench all                # benchmark every model, cache the result
toks --bench qwen3.6:27b-mlx    # benchmark the named model(s)
```

Benchmarks are cached at `~/.cache/toks/tokens-per-second.json`, so plain `toks`
is instant and the TOKENS/S / TTFT columns persist between runs. Rows are sorted
fastest-first; un-benchmarked models sort last.

For Ollama, throughput is computed from the `/api/generate` eval counters. For LM
Studio, `tokens_per_second` and `time_to_first_token` are read straight from the
server's own `stats` — so benchmarking a **remote** LM Studio (e.g. on another host)
stays accurate, since those numbers are measured server-side.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LMSTUDIO_URL` | `http://localhost:1234` | LM Studio endpoint; set to a URL for a remote host, e.g. `http://box.example.net:1234` |
| `LMSTUDIO_API_KEY` / `LM_API_TOKEN` | — | optional `Authorization: Bearer` token |
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

## Behaviour with one backend down

`toks` (provider `all`) lists whatever is reachable and prints a one-line note to
stderr for any configured-but-unreachable backend. It only errors if **no** backend
is reachable. So with LM Studio off, it behaves exactly like the original
Ollama-only tool.

## Tests

```
python3 -m unittest test_toks   # network-free unit tests
```

## Design

See `docs/superpowers/specs/2026-05-31-toks-lmstudio-provider-design.md`.
