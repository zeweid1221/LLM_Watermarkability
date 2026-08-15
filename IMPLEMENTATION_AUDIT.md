# Implementation Audit

This audit covers the code paths used by the paper's composition-pool training,
five-mode generation evaluation, cross-rule analysis, paired PPL analysis, and
teacher-forced ECC diagnostic.

## Verified

- Generic composition training exhausts every 2-, 4-, 7-, and 8-of-16 union
  before repetition and does not receive an evaluated watermark identity.
- Held-out schedules are generated deterministically from fixed seeds and use
  the same 256 LFQA prompts across modes.
- The controller implements the minimum nonnegative uniform target-set shift
  needed to reach pre-temperature mass `tau`.
- PPL is prompt-conditional NLL under the frozen base model and is aggregated
  over equal-length 96-token continuations.
- Cross-rule analysis uses one saved unwatermarked trajectory per model and
  verifies exact prompt and generated-token hashes.
- Paper tables and archived summaries pass their independent value validator.

## Release Fixes

- Prompt tokenization is now identical in trajectory generation, LoRA training,
  held-out generation, and PPL scoring (`add_special_tokens=True`).
- Trajectory caches now include configuration and artifact fingerprints and are
  rejected on any mismatch.
- Partition/model compatibility is checked with a tokenizer-vocabulary hash,
  not only vocabulary size.
- Partition construction excludes special tokens and output-matrix padding rows
  before optimization and handles either left- or right-padded calibration
  batches.
- Adapter loading rejects missing or extra trainable tensors.
- Validation metrics are weighted by active positions rather than averaging
  partial batches equally.
- Scientific settings are stored in scheduler-independent JSON configs. The
  optional Slurm file only requests resources and dispatches a selected config.

## Result Compatibility

Qwen3-8B uses the same prompt token sequence with or without the tokenizer's
special-token flag, so its saved adapter results were unaffected by the
tokenization fix. OPT-125M adds a BOS token, and both the generic
composition-pool run and the schedule-aware appendix control were repeated with
the unified tokenization contract. The released compact tables come from these
corrected runs.

The released partition files are the exact fixed artifacts used by the saved
runs. At runtime, unavailable output-padding rows were excluded and neither
saved model sampled such a row. Rebuilding a partition with the release code
excludes those rows earlier during optimization as well.

## Test Status

The standalone release suite contains 46 passing tests, including a frozen
reference fixture produced by the repository-native KGW implementation.
