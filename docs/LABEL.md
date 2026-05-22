# Labeling Strategy

The dataset under `data/` is generated from the `rcutils` C sources with a deterministic pipeline.

## Extraction

Function definitions are enumerated with `ctags`, then a brace-aware scanner recovers the full source slice for each definition. This keeps the extraction anchored to the actual C source instead of to a hand-maintained list.

## Labeling

- `control_flow_elements` is derived mechanically from the function body by checking for the presence of `if`, `for`, `while`, and `return` as whole words.
- `side_effects` is derived from the function body with a small, fixed rule set that looks for common I/O, memory, and system-clock APIs. If none of those patterns appear, the label is `none`.
- `high_level_purpose` is produced by a deterministic heuristic that uses the function name, common rcutils naming patterns, and a few body-level signals for common cases such as allocation, formatting, environment access, and time conversion.

## Why This Is Reliable Enough

The pipeline is reproducible and low-variance: the same source file always produces the same datapoint. The labels are not free-form guesses over the entire repository; they are constrained by small rule sets and validated after generation. The dataset generator also performs basic sanity checks so malformed extractions or empty labels fail fast.

This is reliable enough for a compact function-level dataset because the goal is consistent structured annotation, not human-level semantic perfection. The extraction is exact, the control-flow and side-effect labels are grounded in lexical evidence from the function body, and the purpose label is intentionally coarse so it stays stable across reruns.