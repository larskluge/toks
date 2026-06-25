# toks

List local LLMs across backends — **Ollama**, **LM Studio**, **mlx-lm**
(`mlx_lm.server`), and **Unsloth Studio** — in one table, ranked by cached
tokens/sec and annotated with effective **bits/weight**. Single file, Python 3
standard library only.

```
$ toks
ollama ✓  lmstudio ✓  mlx ✗  unsloth ✓
PROVIDER │ NAME                        │    SIZE │ TAG  │ PARAMS │  BPW │  CTX │ TTFT │ TOKENS/S │ MODIFIED
─────────┼─────────────────────────────┼─────────┼──────┼────────┼──────┼──────┼──────┼──────────┼────────────
unsloth  │ unsloth/gemma-4-31B-it-GGUF │ 17.1 GB │ gguf │    31B │ 4.78 │ 256k │ 0.10 │     39.6 │ 2 hours ago
ollama   │ gemma4:31b-mlx              │ 18.8 GB │ mlx  │    31B │ 5.18 │ 256k │ 0.37 │     25.0 │ 3 weeks ago
```

The first line is a one-line reachability summary printed to stderr — a green ✓
for each backend that answered the listing request and a red ✗ for one that
didn't (here mlx-lm is down).

The **`BPW`** column is on-disk bits ÷ parameter count — the *effective* width,
which can differ sharply from the quant's nominal one. Above, two 31B Gemma-4
builds tagged the same `31B` separate cleanly: the Q4_K_M GGUF at **4.78** vs the
MLX `nvfp4` build at **5.18** (FP4 weights still carry FP8 block scales). BPW is
exact when the parameter count is known precisely (Ollama, Unsloth's GGUF
header); a leading `~` marks counts estimated from file size. `SIZE` is binary
(GiB), so the GGUF's 18.3 GB on disk shows as `17.1 GB` while `BPW` uses raw bytes.

## Usage

```
toks                            # list models from all reachable backends
toks --provider ollama          # one backend only (ollama | lmstudio | mlx | unsloth | all)
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
over a network, latency leaks into the numbers. Unsloth Studio is a llama.cpp
server, so `toks` reads its server-measured `timings` (`predicted_per_second`,
`prompt_ms`) from the stream — accurate even over a network.

Each cached benchmark records the **observed backend** — the engine inferred from
the response shape (Ollama counters, LM Studio `stats`, llama.cpp `timings`, or a
client-timed mlx_lm stream) plus the server's `system_fingerprint` and the endpoint
actually hit — alongside the configured provider label. Because default ports get
reused (mlx-lm's `:8080` especially), this is what stops a different server that
later binds the port from being silently recorded under the wrong label: if the
observed engine isn't the one the port should host, `toks` warns and stores the
truth, so cross-runtime comparisons stay honest.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LMSTUDIO_URL` | `http://localhost:1234` | LM Studio endpoint; set to a URL for a remote host, e.g. `http://box.example.net:1234` |
| `LMSTUDIO_API_KEY` / `LM_API_TOKEN` | — | optional `Authorization: Bearer` token |
| `MLX_URL` | `http://localhost:8080` | mlx_lm.server endpoint |
| `UNSLOTH_URL` | `http://127.0.0.1:8888` | Unsloth Studio endpoint |
| `UNSLOTH_API_KEY` | — | `Authorization: Bearer` token for Unsloth Studio (required) |
| `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | HF cache used for mlx / Unsloth (GGUF) metadata enrichment |
| `XDG_CACHE_HOME` | `~/.cache` | cache location root |
| `XDG_CONFIG_HOME` | `~/.config` | location of the optional `.env` (see below) |

A bare host (no scheme) is accepted — `http://` is assumed.

On startup `toks` reads `~/.config/toks/.env` (honoring `XDG_CONFIG_HOME`) for any
of the variables above — handy for the `UNSLOTH_API_KEY`. It is **read-only**: a
real environment variable always wins, and `toks` never writes the file. Lines are
`KEY=VALUE` (blank lines and `#` comments ignored; a leading `export` and
surrounding quotes are stripped).

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

## Unsloth Studio prerequisites

1. Start Unsloth Studio (default port `8888`) and put its key in
   `~/.config/toks/.env` as `UNSLOTH_API_KEY=…` (every API route is auth-gated).
2. Listing uses `GET /api/hub/local` (Studio's downloaded-model inventory), **not**
   `/v1/models` — the latter reports only the model currently *loaded* for serving,
   so it is empty whenever nothing is loaded. `toks` keeps the HF-cache,
   text-generation entries (dropping image/embedding/speech repos and the LM
   Studio / Ollama models that toks' own providers already list). The listing is
   format-agnostic: GGUF **and** MLX/safetensors Unsloth weights both appear.
3. The list is **not** scoped to `unsloth/*`. An `mlx-community/*` chat model that
   also sits in the HF cache is listed under `unsloth` on purpose, so Studio's
   throughput for the exact same files can be compared against the `mlx` provider's
   number for them.
4. Rows are enriched from the local **HF cache** by parsing the model's main GGUF
   header — exact parameter count (summed from the tensor table), quant
   (`general.file_type`), context length, MoE shape. This works only when `toks`
   runs on the server's machine; for a remote `UNSLOTH_URL`, and for MLX/safetensors
   models (no GGUF header), those columns show `-` (the benchmark still works).
5. A repo may ship extra GGUFs (`mmproj-*` vision projectors, an `mtp-*` draft
   head); `toks` sizes and parses the main weights file, ignoring those.
6. Models are usually already resident; the usual 1-token warmup absorbs any load.

## Behaviour with one backend down

`toks` (provider `all`) lists whatever is reachable and prints a single
reachability line to stderr — each backend with a green ✓ (answered) or red ✗
(unreachable), no error detail. Colour is emitted only when stderr is a TTY and
`NO_COLOR` is unset. It only errors if **no** backend is reachable. So with LM
Studio and mlx-lm off, it behaves exactly like the original Ollama-only tool.

## Tests

```
python3 -m unittest test_toks   # network-free unit tests
```

## Design

See `docs/superpowers/specs/2026-05-31-toks-lmstudio-provider-design.md`,
`docs/superpowers/specs/2026-06-06-mlx-provider-design.md`,
`docs/specs/2026-06-24-unsloth-studio-provider-and-bpw-design.md`, and
`docs/specs/2026-06-25-unsloth-local-listing-and-bench-backend-identity-design.md`.
