# toks: loaded-model indicator (green ✓) — design

Date: 2026-07-09

## Summary

Show, in the `toks` listing, which model(s) are **currently loaded** (in memory /
serving) as opposed to merely downloaded. The marker is a **green ✓ prefixed
inside the `NAME` column** — no new column, no new separator, so the table stays
as narrow as it is today. A loaded row reads `✓ <name>`; an unloaded row reads
`  <name>` (two leading spaces, so names stay aligned). The ✓ reuses the same
spring-green (`STATUS_OK`, `38;5;78`) already used by the reachability status
line, and is emitted colorised only when stdout is a TTY and `NO_COLOR` is unset.

```
PROVIDER │ NAME               │  SIZE │ TOKENS/S
─────────┼────────────────────┼───────┼─────────
unsloth  │ ✓ gemma-4-31B-GGUF │ 17 GB │     39.6
ollama   │   gemma4:31b-mlx   │ 18 GB │     25.0
ollama   │ ✓ qwen3:8b         │  5 GB │     61.2
```

"Loaded" is backend-specific and, for two providers, needs one extra cheap
localhost call; the detection rules and their sources are the core of this design.

## Goals

- Add a per-record `loaded: bool` flag and populate it correctly per provider.
- Render a green ✓ as a 2-column prefix inside the `NAME` cell for loaded rows,
  aligned so unloaded names are indented by the same 2 columns.
- Make `table()` ANSI-aware so a colourised cell no longer breaks column width /
  alignment (currently every cell is plain text, so `len()` == visible width; the
  ✓ is the first coloured cell content).
- Colour the ✓ only when stdout is a TTY and `NO_COLOR` is unset (same rule the
  status line already applies to stderr) — piping `toks` stays clean and greppable.
- Keep the single-file, standard-library-only constraint and the per-provider
  class + parser pattern.

## Non-goals / out of scope

- **mlx.** `mlx_lm.server`'s `/v1/models` scans the HF cache and lists *every*
  candidate model, with no `/ps`-style endpoint reporting which one is resident.
  There is no reliable signal, so mlx rows are **never marked** (a documented
  limitation) rather than risk false positives. Revisit if mlx_lm gains a
  loaded-state endpoint.
- Sorting, filtering, or a `--loaded`-only flag. Purely visual for now (YAGNI).
- A legend / header text for the ✓. The green tick is self-evident and adding a
  header defeats the space-saving intent. The reachability line already trains the
  reader that green ✓ = good/present.
- Any change to benchmarking, caching, or the other columns.

## Current state (relevant pieces)

- `ModelRecord` (`toks:67`) has no notion of "loaded". `ctx_loaded` is set by two
  providers but means "running context length", not a boolean load state.
- `build_rows(records, cache)` (`toks:1273`) builds the string grid; the `NAME`
  cell is `rec.name or "-"`. `table(rows)` (`toks:1331`) computes each column
  width with `len(cell)` and pads with `str.ljust/rjust` — both count raw
  characters, so any embedded ANSI escape would corrupt alignment.
- Colour for the **status line** is decided in `main()` (`toks:2016`) as
  `sys.stderr.isatty() and not os.environ.get("NO_COLOR")`. The **table** (stdout)
  is printed with no colour today.
- `_color(text, code, enabled)` (`toks:1314`) and `STATUS_OK` (`toks:1310`)
  already exist and are exactly what the ✓ needs.
- `_canonical_model_id(value)` (`toks:992`) reduces any model id — including an
  HF-cache path — to a comparable lower-cased `org/name`; reused for Studio
  matching below.

## Design

### 1. Data model

Add one field to `ModelRecord`:

```python
loaded: bool = False   # True when the model is resident/serving, not just on disk
```

Default `False` keeps every existing construction site and test valid.

### 2. Detection per provider

| Provider | Source of "loaded" | Extra call? |
|----------|--------------------|-------------|
| **lmstudio** | `state == "loaded"` in the `/api/v0/models` item (fallback: `loaded_context_length` present) | no — already fetched |
| **llama** | `/v1/models` lists only the one served model → every llama record is loaded | no |
| **ollama** | model appears in `/api/ps` (matched by `digest`, fallback `name`) | +1 (`/api/ps`) |
| **unsloth** | model id equals the one `/v1/models` reports as loaded (canonicalised match) | +1 (`/v1/models`) |
| **mlx** | not detectable — never marked | no |

Details:

- **lmstudio** — in `lmstudio_parse_models` (`toks:125`), set
  `loaded = (item.get("state") == "loaded") or item.get("loaded_context_length") is not None`.
  Zero cost: the field rides on the listing already parsed.

- **llama** — in `llama_parse_models` (`toks:918`), set `loaded=True` on every
  record. `llama-server` runs one model per process and `/v1/models` names exactly
  it; if a future build ever lists several, they are all the resident set by the
  same contract, so marking all is still correct.

- **ollama** — add a parser `ollama_running_digests(ps_payload) -> tuple[set, set]`
  returning `(digests, names)` from `/api/ps` `models[]` (`digest`, `name`).
  `OllamaProvider.list_models` (`toks:1521`) makes one more `GET /api/ps` (wrapped
  in try/except → empty sets on failure, so a missing/old endpoint degrades to "no
  marks" and never breaks the listing), then sets
  `rec.loaded = rec.digest in digests or rec.name in names`. Digest is the robust
  key (both `/api/tags` and `/api/ps` carry it); name is the fallback.

- **unsloth** — add `unsloth_served_ids(models_payload) -> set` returning the
  canonicalised ids from `/v1/models` `data[].id`. `UnslothProvider.list_models`
  (`toks:1631`) makes one more `GET /v1/models` (try/except → empty set; Studio may
  have nothing loaded, which 200s-empty or errors — both mean "no marks"), then
  sets `rec.loaded` when `_canonical_model_id(rec.name)` is in that set, matching
  on the full `org/name` and on the bare alias (last path segment), mirroring
  `served_model_matches`.

The two extra calls are localhost, sub-listing-sized, and issued once per run — no
measurable latency next to the existing `/api/show` fan-out.

### 3. Rendering

- `build_rows` gains a `color: bool` parameter. For each record it prefixes the
  `NAME` cell with a 2-column marker:

  ```python
  mark = _color("✓", STATUS_OK, color) if rec.loaded else " "
  name_cell = f"{mark} {rec.name or '-'}"
  ```

  The header row keeps a bare `"NAME"` (matches the approved mockup: `NAME` sits
  over the tick column, names indented 2). Unloaded rows use a plain space so the
  visible prefix is always exactly 2 columns and names align.

- `table()` becomes ANSI-aware. Add:

  ```python
  _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
  def _visible_width(s): return len(_ANSI_RE.sub("", s))
  ```

  Width computation uses `_visible_width(cell)`; padding uses a small helper that
  appends/prepends spaces based on `width - _visible_width(cell)` (since
  `str.ljust/rjust` would count the escape bytes). The separator line already uses
  the numeric widths, so it is unaffected. Requires a new `import re` (stdlib —
  the no-third-party-deps constraint is unchanged).

### 4. Colour flag

In `main()` (`toks:2035`), compute the table's colour the same way the status line
does for stderr, but against stdout:

```python
table_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
print(table(build_rows(display, cache, color=table_color)))
```

Piped/redirected output gets a plain `✓ ` prefix (still a meaningful, greppable
marker); a TTY gets the green one.

## Testing

`test_toks.py` is stdlib `unittest`. Add:

- **Detection, per provider (parser-level, no network):**
  - lmstudio: `state: "loaded"` → `loaded True`; `"not-loaded"` / absent → `False`;
    `loaded_context_length` present with no `state` → `True`.
  - llama: any `/v1/models` record → `loaded True`.
  - ollama: `ollama_running_digests` extracts digests+names; a record whose digest
    matches → `True`; digest mismatch but name match → `True`; neither → `False`;
    empty/malformed ps payload → all `False`.
  - unsloth: `unsloth_served_ids` canonicalises `/v1/models` ids (incl. an HF-cache
    path); record matching by full id and by bare alias → `True`; no match → `False`;
    empty payload → `False`.
- **Rendering:**
  - `build_rows(..., color=False)` prefixes `"✓ "` for a loaded record and `"  "`
    for an unloaded one; header `NAME` unchanged; both name columns align.
  - `build_rows(..., color=True)` wraps the ✓ in the `STATUS_OK` escape.
  - `table()` alignment holds with a colourised ✓ present: the column width and
    every row's separator position match the plain-text case (regression guard for
    the ANSI-aware width).
  - `_visible_width` returns the escape-stripped length.

## Limitations

- **mlx** models are never marked (no loaded-state signal from `mlx_lm.server`).
- Detection is a snapshot at listing time; a model loaded/evicted between the
  listing call and reading is not reflected (acceptable — the whole listing is a
  snapshot).

## Files touched

- `toks` — `ModelRecord` (+`loaded`); `lmstudio_parse_models`, `llama_parse_models`
  (+detection); new `ollama_running_digests`, `unsloth_served_ids`;
  `OllamaProvider.list_models`, `UnslothProvider.list_models` (+extra call);
  `build_rows` (+`color`, marker prefix); `table` + `_visible_width` (ANSI-aware);
  `main` (table colour flag); `import re`.
- `test_toks.py` — detection + rendering tests above.
- `README.md` — note the green ✓ loaded marker (and the mlx caveat) alongside the
  existing status-line description.
