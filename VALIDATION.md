# Bundle validation

Validated locally on August 10, 2026 with OPT-125M and an RTX 4060 Laptop GPU.

- Python compilation passed for all five Python entry points.
- The target-set unit smoke verified the closed-form mass equation, the
  10-word VT-Hamming codebook, and 7+1 block schedules.
- The packaged manifest contains 5,756 unique prompts: 5,000 train, 500
  validation, and 256 held-out test rows.
- Manifest SHA-256:
  `99538c8bd6f0de4c1e598e9c8d67df88027ec48e2d4fd7294080eadc62efd56f`.
- A full two-example GPU smoke completed hard trajectory generation, one LoRA
  epoch, validation, checkpoint reload, and all four generation/evaluation
  modes with finite outputs.

The smoke checks implementation plumbing only. Its two-example metrics are not
scientific results and are excluded from the MSI archive.
