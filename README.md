# LLM Watermarkability

LLM Watermarkability studies how readily a pretrained language model can realize externally specified watermark target sets. The current implementation combines a target-conditioned low-rank adapter with a closed-form residual controller: the adapter reshapes the model distribution toward the requested token set, while the controller supplies only the remaining bias required to reach a chosen adherence target.

This repository is an isolated research implementation. It does not modify or depend on the ECC-IW detector repository.

## Components

- `modeling.py`: custom LoRA layers, target-conditioned projection, target-set summaries, and the closed-form residual controller.
- `targets.py`: static and dynamic target rules for ECC, binary partitions, Combinatorial Watermarking, Unigram, KGW, atomic KGW, and entropy-gated KGW.
- `watermarkability/`: hierarchical partition representation and probability-balanced atomic partition optimization.
- `run_experiment.py`: trajectory preparation, adapter training, controlled generation, and held-out evaluation.
- `build_atomic16_partition.py`: construction of a reusable 16-atom vocabulary partition from model probabilities and token embeddings.
- `compare_*.py` and `analyze_insample_generalization.py`: capacity, objective, transfer, and in-sample diagnostics.
- `test_*.py`: objective, static/dynamic target, and multi-rule regression tests.
- `*.slurm` and `submit_*.sh`: MSI launchers for the OPT-125M and Qwen3-8B experiments.

## Current Protocol

The latest experiment trains one rank-16 target-conditioned adapter over seven target families:

```text
ecc, binary, cw_abcd, unigram,
atomic_kgw, atomic_kgw25, entropy_kgw
```

Each family uses 5,000 training prompts and 500 validation prompts. Evaluation compares unwatermarked generation, fixed logit bias, the closed-form controller, LoRA alone, and LoRA plus the residual controller on held-out prompts. Generated datasets, partitions, checkpoints, and result archives are intentionally excluded from version control.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the mixed prompt manifest from public datasets:

```bash
python build_manifest.py \
  --output data/mixed_prompts_5756.jsonl \
  --train-count 5000 \
  --validation-count 500 \
  --test-count 256
```

The atomic-16 partition additionally requires the source model tokenizer and embeddings. Its CLI documents all model, manifest, and output paths:

```bash
python build_atomic16_partition.py --help
```

## Tests

```bash
python -m unittest \
  test_objectives \
  test_dynamic_targets \
  test_mixed_objective \
  test_atomic16_targets -v
```

## MSI

The launchers assume a CUDA environment and write all generated artifacts under ignored `data/`, `assets/`, `outputs/`, and `logs/` directories. For the latest seven-rule run:

```bash
bash submit_opt_atomic16_multirule.sh
```

The Qwen3-8B A100 pipeline builds its model-specific atomic partition and then
submits the seven-rule experiment with an `afterok` dependency:

```bash
bash submit_qwen3_atomic16_a100_pipeline.sh
```

The Qwen3 smoke and runtime launchers can be submitted independently before a
full run:

```bash
sbatch run_qwen3_a100_smoke.slurm
sbatch run_qwen3_a100_runtime_benchmark.slurm
```

After completion:

```bash
python validate_atomic16_pipeline.py \
  --partition-dir assets/atomic16/opt-125m-lfqa5000-v1 \
  --experiment-dir outputs/opt125m_atomic16_universal_7rules_lfqa5000_each_rank16_epochs6_seed20260811 \
  --expected-rules ecc,binary,cw_abcd,unigram,atomic_kgw,atomic_kgw25,entropy_kgw
```

## Status

The implementation supports controlled OPT-125M ablations and the corresponding
Qwen3-8B A100 pipeline. The same abstractions cover static and prefix-dependent
target sets.
