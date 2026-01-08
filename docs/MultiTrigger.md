Multi-trigger (CSV) support — Usage Examples

This document shows example configs and README notes for using CSV trigger → DOP class mapping.

Example single-trigger (unchanged behavior):

train:
  trigger: "Jinx"
  diff_output_preservation: true
  diff_output_preservation_class: "Woman"

Example multiple trigger → class mapping (subject + props):

train:
  trigger: "Jinx, Zapper"
  diff_output_preservation: true
  diff_output_preservation_class: "Woman, Gun"

Behavior:
- All occurrences of "Jinx" are replaced with "Woman" and all occurrences of "Zapper" replaced with "Gun" during DOP precompute and runtime DOP generation.
- If the trigger list is longer than the class list, missing classes are treated as blank (removed from captions).

Caveats & notes:
- Matching is exact and case-sensitive by default ("Jinx" != "jinx").
- Whitespace is normalized before replacement to reduce tokenizer merging issues. If you need case-insensitive matching or more advanced token-aware behavior, open a follow-up issue/PR.
- `photomaker_pipeline` still enforces single special token for compatibility; relaxing that is a higher-risk follow-up.
