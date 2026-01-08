# Multi-trigger (CSV) support for DOP — Design & Plan ✅

**Goal:** Add support for multiple trigger tokens mapped to multiple Differential Output Preservation (DOP) classes using a comma-separated value (CSV) syntax while remaining 100% backwards compatible with the current single-token behavior.

This file outlines the specification, pragmatic implementation steps, test cases, docs to update, and rollout plan. The chosen approach is minimal-risk: parse CSVs, build an ordered (trigger → class) mapping, and perform deterministic pairwise replacements during DOP precompute and anywhere else a single-trigger replacement currently happens.

---

## Overview & Motivation 💡

- Problem: Training subject LoRAs often requires learning both a subject and their props (e.g., `Jinx` and `Zapper`). Current DOP and trigger handling implicitly assume a single trigger and a single DOP class.
- Proposal: Interpret `train_config.trigger` and `train_config.diff_output_preservation_class` as comma-separated lists of equal-ordered values. Replace each trigger value with the mapped DOP class value (empty if missing) during DOP precompute and related places.
- Backwards compatible: A single value behaves exactly as today.

---

## Config syntax & semantics

- `trigger`: either a string (current single value) OR CSV string like `"Jinx, Zapper, Vest"`.
- `diff_output_preservation_class`: either a string OR CSV string like `"Woman, Gun, "` (note trailing empty class allowed).

Rules:
- Trim whitespace around values.
- If the triggers list is longer than the classes list, missing classes are treated as empty string (i.e., replacements remove the trigger text).
- If classes list is longer than triggers list, extra classes are ignored.
- Matching defaults to **exact, case-sensitive** string matching on the caption text (config option can be added for case-insensitive matching later).

---

## Replacement semantics (algorithm)

1. Parse `trigger_csv = parse_csv_list(train_config.trigger)` and `class_csv = parse_csv_list(train_config.diff_output_preservation_class)`.
2. Build ordered pairs: pairs = [(t_i, class_csv[i] if i < len(class_csv) else "")].
3. Normalize caption text (small helper): ensure consistent separators (spaces after commas), collapse duplicate spaces, and strip leading/trailing punctuation around tokens where meaningful. This normalization avoids tokenizer merges in many common cases.
4. Sort replacement pairs by length of trigger (descending) to avoid substring collisions (e.g., `"jinx master"` before `"jinx"`).
5. For each (trigger, cls) in pairs:
   - `escaped_trigger = re.escape(trigger)`
   - Perform a replace that uses word boundaries where possible: `re.sub(rf"(?<!\S){escaped_trigger}(?!\S)", cls, caption)`
   - Fall back to plain replace if word-boundary matching fails due to punctuation layout.
6. Replace all occurrences (global) of each trigger.

Notes:
- Use `re.escape` to avoid regex injection.
- Word-boundary matching protects against replacing substrings inside words; in languages with different tokenization this is a reasonable safe default.

---

## Implementation tasks (files & code-level changes) 🔧

Priority: Minimal, low-risk, and fully backward compatible.

1. Add utility functions
   - File: `toolkit/prompt_utils.py`
     - Add: `def parse_csv_list(value: Optional[str]) -> List[str]:` — returns list of trimmed, non-empty tokens (unless empty entry allowed for explicit blank class), keep empty strings if user explicitly includes them (e.g., `"Woman, "`).
     - Add: `def normalize_caption_separators(text: str) -> str:` — small normalizer to ensure space after commas and collapse repeated spaces; optional param to preserve punctuation.

2. Update SDTrainer DOP precompute
   - File: `extensions_built_in/sd_trainer/SDTrainer.py`
     - Location: where DOP currently runs a single replace (search for `diff_output_preservation_class` or the existing replace code).
     - Replace single-value logic with CSV parsing + pairwise replacements using the algorithm above.
     - Add logging.warn when len(triggers) != len(classes) to explain blanks will be used (non-fatal).
     - Ensure the new code runs only when DOP is enabled and preserve the same caching keys so existing caches still match when only single values used.

3. Make dataset-level triggers explicit
   - File: `toolkit/dataloader_mixins.py` or the dataset config loader
     - Ensure any appended triggers (`random_triggers` etc.) are included in the trainer-level canonical trigger list in a deterministic order (note: appended triggers can be treated as appended to the `trigger` list, or handled separately — recommend canonicalizing into a single list for replacements).

4. Photomaker / pipeline note (follow-up)
   - File: `toolkit/photomaker_pipeline.py`
     - Current: rejects multiple trigger tokens. Leave untouched in first pass; add tests and document behavior. Decide in follow-up whether to relax this restriction.

5. Add tests
   - File: `testing/test_dop_multi_trigger.py`
     - Cases (see next section).

6. Docs & examples
   - Update: `README.md`, `config/examples/` and `docs/` with examples demonstrating CSV mapping.

---

## Test matrix / unit tests ✅

Create a dedicated test file `testing/test_dop_multi_trigger.py` with the following tests:

- test_single_trigger_unchanged_behavior
  - Input: trigger=`"Jinx"`, class=`"Woman"`, caption=`"A Jinx portrait"` -> assert replacement -> `"A Woman portrait"` and existing DOP paths unchanged.

- test_multiple_triggers_pairwise_mapping
  - Input: triggers=`"Jinx, Zapper"`, classes=`"Woman, Gun"`, caption=`"Jinx with a Zapper"` -> `"Woman with a Gun"`.

- test_more_triggers_than_classes
  - Input: triggers=`"Jinx, Zapper, Vest"`, classes=`"Woman, Gun"`, caption=`"Jinx Zapper Vest"` -> `"Woman Gun "` (note trailing blank for Vest).

- test_whitespace_and_commas
  - Input: inconsistent spacing: `"Jinx,Zapper ,  Vest"` -> ensure parsing normalizes to `['Jinx','Zapper','Vest']`.

- test_repeated_occurrences
  - Caption with multiple occurrences of a trigger -> all occurrences replaced.

- test_overlapping_triggers_longest_first
  - triggers=`"Jinx, Jinx Master"`, classes=`"Woman, Veteran"`, caption=`"Jinx Master and Jinx"` -> ensure `"Veteran and Woman"` (longest-first ensures `"Jinx Master"` replaced before `"Jinx"`).

- test_case_sensitivity_default
  - By default, `"jinx"` should NOT match `"Jinx"` unless case-insensitive option added.

- test_dop_cache_key_consistency
  - Verify that for single-item config, the precompute cache key is identical to current (no cache invalidation for existing single-trigger use).

- test_photomaker_rejection_logged
  - When multiple triggers are present, photomaker still raises/ logs; confirm behavior and document.

Add tests for config parsing edge cases (empty strings, trailing commas, explicit empty class tokens etc.).

---

## Backwards compatibility & migration notes 🔁

- Behavior for single tokens is unchanged.
- Existing cached DOP precompute artifacts for single tokens should remain valid; ensure cache key derivation uses normalized single-token canonical form.
- If users supply CSV values and expect case-insensitive matching, call out that we default to case-sensitive; provide a config flag `trigger_case_insensitive` in a separate change if needed.

---

## Logging & UX

- Warn (not fail) when `len(triggers) != len(classes)` with message: `"Trigger list length != DOP class list length: missing classes will be replaced with empty strings. Provide explicit blanks if you want empty replacement: 'Woman, , Gun'"`.
- Warn (not fail) when multiple trigger instances are detected in a single caption (helps users discover unexpected duplicates).
- Add a short section in README `Trigger & DOP CSV Support` with examples and a note about `photomaker_pipeline` behavior being unchanged for now.

---

## Risk & Caveats ⚠️

- Tokenization/punctuation edge cases exist (e.g., languages or models with non-space tokenization). The normalize helper reduces many common issues but cannot 100% guarantee correct token boundaries across all tokenizers.
- If downstream pipelines (e.g., `photomaker_pipeline`) enforce a single special token, relaxing that behavior is more invasive and should be done in a separate PR with additional tests.
- Overly permissive replacement (e.g., not using word boundaries) could replace substrings inside words — algorithm uses word boundaries by default to avoid that.

---

## Rollout plan & timeline 🗓️

1. Implement utilities and CSV parsing (small PR) — ~1–2 hrs.
2. Implement DOP precompute replacements + logging + normalization (small PR) — ~2–4 hrs.
3. Add unit tests for all cases above — ~1–2 hrs.
4. Run test suite and address failing DOP tests — ~1 hr.
5. Update README and example configs; add a short example in `config/examples/` — ~30–60 mins.
6. Consider follow-up PR to relax `photomaker_pipeline` constraints if desired — separate plan.

---

## Example pseudocode

```py
# toolkit/prompt_utils.py
import re

def parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(',')]
    # Keep empty strings if explicitly present ("a, , c")
    return parts


def normalize_caption_separators(text: str) -> str:
    # simple: ensure space after comma, collapse repeated spaces
    text = re.sub(r",\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# In SDTrainer (DOP precompute):
triggers = parse_csv_list(train_config.trigger)
classes = parse_csv_list(train_config.diff_output_preservation_class)
# build pairs (missing class -> '')
pairs = [(t, classes[i] if i < len(classes) else '') for i, t in enumerate(triggers)]
# sort by length desc
pairs.sort(key=lambda tcls: len(tcls[0]), reverse=True)

caption = normalize_caption_separators(caption)
for trigger, cls in pairs:
    if trigger == '':
        continue
    esc = re.escape(trigger)
    # try a word-boundary replace
    pattern = rf"(?<!\S){esc}(?!\S)"
    caption, n = re.subn(pattern, cls, caption)
    if n == 0:
        # fallback to plain replace
        caption = caption.replace(trigger, cls)
```

---

## Open questions for review

- Should we add a `trigger_case_insensitive` config option now or handle it later? (recommend defer)
- How should we surface examples for multi-trigger usage in the UI/README? (add simple `config/examples/multi_trigger.yaml`)
- Should `photomaker_pipeline` allow multiple triggers simultaneously? (defer to follow-up PR)

---

## Summary ✅

This plan introduces minimal, backward-compatible behavior to support subject+prop learning using simple CSV mapping for `trigger` → `DOP class`. It keeps the default behavior intact, reduces the risk of breaking existing caches, and adds comprehensive tests and documentation. The implementation has been completed for the core path (CSV parsing helpers, SDTrainer DOP precompute and runtime mapping, and unit tests). Follow-up tasks include relaxing `photomaker_pipeline`'s single-trigger constraint and adding UI docs.

---

## Implementation status (completed)
- CSV parsing & caption normalization utilities added to `toolkit/prompt_utils.py` ✅
- SDTrainer: CSV-based trigger→class mapping implemented for DOP precompute and runtime embedding generation ✅
- Unit tests added: `testing/test_dop_multi_trigger.py` (covers parsing, mapping, overlapping triggers, mismatched lengths) ✅
- Documentation added: `docs/MultiTrigger.md` and this plan updated to reflect changes ✅

---

If you'd like, I can now:
- Open a focused PR that includes these changes and add the `LEARNINGS.md` entry, or
- Proceed to relax `photomaker_pipeline` (higher-risk) and add integration tests for end-to-end DOP caching behavior.

Which do you want me to do next?