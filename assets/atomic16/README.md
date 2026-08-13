# Released Atomic Partitions

These directories contain the exact fixed model-specific partitions used by the
saved OPT-125M and Qwen3-8B paper runs. Each partition includes:

- token-to-atom and semantic-group arrays;
- banned-token and ECC-boundary token IDs;
- the selected ECC, Binary, and CW atom compositions;
- tokenizer identity metadata and partition diagnostics.

`run_experiment.py` verifies the tokenizer vocabulary SHA-256 before using a
partition. Rebuild the artifact with `build_atomic16_partition.py` when changing
the model or tokenizer.
