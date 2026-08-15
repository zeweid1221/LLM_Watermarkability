# LLM Watermarkability

This repository contains the implementation and experiment drivers for **LLM
Watermarkability**. The framework accepts an externally defined token target set
at each decoding step, uses a target-conditioned LoRA adapter to make that set
more compatible with the model distribution, and applies a closed-form residual
controller to reach a requested target mass.

The watermark schedule and detector remain external. The implementation covers
the paper's ECC, Binary, CW-ABCD, Unigram, Atomic-KGW50, Atomic-KGW25, and
entropy-gated Atomic-KGW schedules over a shared 16-atom vocabulary partition.

## Paper Experiment Map

| Paper experiment | Entry point | Validation |
|---|---|---|
| Composition-pool LoRA training and five-mode evaluation | `run_experiment.py` | `validate_atomic16_pipeline.py` |
| Cross-rule watermarkability on shared unwatermarked trajectories | `analyze_cross_rule_watermarkability.py` | `validate_cross_rule_watermarkability.py` |
| Paired PPL reductions and bootstrap intervals | `analyze_composition_pool_paired.py` | paper result validator |
| Teacher-forced ECC diagnostic | `analyze_teacher_forced_watermarkability.py` | `test_teacher_forced_watermarkability.py` |
| Corrected OPT schedule-aware appendix control | `configs/opt_schedule_aware.json` | paired bootstrap and paper result validator |

`modeling.py` implements LoRA conditioning and the analytic controller;
`targets.py` implements target schedules and generic atom unions; and
`watermarkability/` builds and stores model-specific atomic partitions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The released `assets/atomic16/` directories are the exact fixed partitions used
by the saved paper runs. They are small enough to version and avoid rebuilding a
model-dependent preprocessing artifact.

## Data

Build the deterministic 5,000/500/256 LFQA manifest from the public LFQA train
split:

```bash
python build_lfqa_manifest.py \
  --output data/lfqa_prompts_5756.jsonl \
  --train-count 5000 \
  --validation-count 500 \
  --test-count 256 \
  --seed 20260810
```

The builder uses cached Arrow shards when available and otherwise downloads
`vblagoje/lfqa`. `run_experiment.py` rejects empty prompts, unknown splits, and
duplicate prompts across splits.

To rebuild a partition rather than use the released artifact:

```bash
python build_atomic16_partition.py \
  --model-name facebook/opt-125m \
  --manifest data/lfqa_prompts_5756.jsonl \
  --output-dir assets/atomic16/opt-125m-lfqa5000-v1 \
  --device cuda
```

The builder excludes tokenizer special IDs and output-matrix padding rows. It no
longer requires a partition from another repository.

## Reproducing Paper Runs

The canonical paper protocol trains on all
2-, 4-, 7-, and 8-of-16 atom unions without invoking an evaluated watermark
schedule. With 5,000 prompts repeated four times, it uses 20,000 training and
2,000 validation trajectories. Watermark schedules are attached only on the 256
held-out prompts.

Scientific settings are stored in scheduler-independent JSON files:

| Config | Paper role |
|---|---|
| `configs/opt_composition_pool.json` | Main OPT composition-pool result |
| `configs/qwen3_composition_pool.json` | Main Qwen3 composition-pool result |
| `configs/opt_schedule_aware.json` | Corrected OPT schedule-aware appendix control |
| `configs/qwen3_schedule_aware.json` | Qwen3 schedule-aware appendix control |

Run any protocol locally or inside an allocated GPU job:

```bash
python run_paper_protocol.py --config configs/opt_composition_pool.json
```

For Slurm, the repository provides one optional scheduler wrapper. Cluster
partition, GPU type, memory, and time directives may need adjustment:

```bash
export WM_LORA_ROOT="$HOME/LLM_Watermarkability"
sbatch scripts/hpc/run_config.slurm configs/opt_composition_pool.json
```

The wrapper contains no scientific hyperparameters; it only allocates compute
and invokes the selected JSON config. The schedule-aware controls use the same
held-out prompts and evaluation settings as their generic counterparts while
training on 5,000 prompts for each of seven request schedules. The released OPT
table comes from the corrected special-token-consistent rerun.

The shared-trajectory analysis is scheduler independent as well. Run
`python analyze_cross_rule_watermarkability.py --help` for model-specific
arguments, then combine the OPT and Qwen outputs with
`validate_cross_rule_watermarkability.py`.

## Validation

Run the complete CPU test suite:

```bash
python -m unittest discover -p "test_*.py" -v
```

Validate a completed main run:

```bash
python validate_atomic16_pipeline.py \
  --partition-dir assets/atomic16/opt-125m-lfqa5000-v1 \
  --experiment-dir outputs/opt125m_composition_pool_release_lfqa5000x4_rank16_epochs6_seed20260810 \
  --expected-rules ecc,binary,cw_abcd,unigram,atomic_kgw,atomic_kgw25,entropy_kgw
```

Trajectory caches carry hashes of the manifest, ordered prompts, target arrays,
partition files, tokenizer, model, and generation settings. A stale cache is
rejected rather than silently reused. Adapter checkpoints are also loaded with
an exact trainable-key check.

See `IMPLEMENTATION_AUDIT.md` for the release audit. The compact aggregate
tables used by the paper and their independent numeric validator are released
under `paper_tmlr/data/` and
`paper_tmlr/scripts/validate_results.py`; model checkpoints and token-level
generation archives remain excluded because of their size.
