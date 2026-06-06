# toks CLI flags cleanup

**Date:** 2026-06-06
**Status:** Approved

## Goal

Shrink the flag surface from seven flags to two. `--bench` absorbs the
mode/selection flags; the benchmark tuning flags go away.

## New CLI surface

```
toks                          # list models (unchanged)
toks --provider ollama        # ollama | lmstudio | all (unchanged)
toks --bench                  # benchmark models missing a cached value (default)
toks --bench missing          # same, explicit
toks --bench all              # benchmark every listed model
toks --bench MODEL [MODEL...] # benchmark the named model(s)
```

## Changes

### `--bench` takes optional targets

- `nargs='*'`, default `None` (no benchmarking).
- Empty list (`toks --bench`) or `missing` → **missing** mode: benchmark only
  models with no cached tokens/sec value.
- `all` → benchmark every listed model.
- Any other value(s) → treat as model names; benchmark exactly those.
- `all` or `missing` combined with anything else is a parser error.
- A model literally named `all` or `missing` is shadowed by the keywords;
  accepted limitation.

### Removed flags

| Flag | Replacement |
|---|---|
| `--all` | `--bench all` |
| `--missing` | `--bench` (now the default mode) |
| `--model NAME` | `--bench NAME` (benchmarking); table filtering dropped — grep instead |
| `--prompt` | none; `DEFAULT_PROMPT` constant |
| `--max-tokens` | none; `DEFAULT_MAX_TOKENS` constant |

### Behavior notes

- `--bench MODEL` benchmarks the named models, then prints the **full**
  table (the old `--model` also filtered the display; that's gone).
- Unknown model names still warn on stderr (`warning: model not found: ...`).
- Missing mode still skips models with a cached value; non-benchmarkable
  models still print the `skipping ...: unsupported model type/format` note.
- Cache entries keep recording `prompt`/`max_tokens` (now always the
  defaults), so the cache schema is unchanged.

## Implementation shape

- Extract argument parsing into `parse_args(argv)` returning the parsed
  namespace, so target validation is unit-testable.
- Add a small target-selection step that maps `(records, targets, cache)` →
  records to benchmark, replacing the `args.all/args.missing/args.model`
  branching in `main()`/`run_benchmarks()`.
- `run_benchmarks()` uses `DEFAULT_PROMPT`/`DEFAULT_MAX_TOKENS` directly
  instead of reading them off `args`.
- Update README usage block.
- Unit tests: parser accepts/rejects the new shapes; target selection picks
  missing/all/named records correctly.

## Out of scope

- No changes to providers, cache format, table rendering, or sorting.
