# Watermarkability

This repository contains the implementation and experiment drivers for
**Watermarkability: Measuring and Improving Language-Model Compatibility with
Logit-Based Watermarks**.

Paper: [arXiv:2610.22810](https://arxiv.org/abs/2610.22810)

The framework measures how naturally a language model supports a requested
token set at each decoding step. It also trains a target-conditioned LoRA
adapter to improve that compatibility, then applies a closed-form residual
controller to reach the requested target mass. Watermark schedules and
detectors remain external to the adapter.

The paper evaluates six primary schedules: ECC, Binary, CW-ABCD, Unigram,
Grouped-KGW50, and Grouped-KGW25. Entropy-gated KGW is retained as an auxiliary
appendix schedule. All schedules are represented through the same 16-base-bucket
partition. For compatibility with saved runs, the code and CSV files retain the
legacy IDs `atomic_kgw` and `atomic_kgw25` for Grouped-KGW50 and
Grouped-KGW25.

## Repository Map

| Component | Purpose |
|---|---|
| `modeling.py` | Target-conditioned LoRA and analytic residual controller |
| `targets.py` | Target schedules and generic base-bucket unions |
| `watermarkability/` | Model-specific partition construction and optimization |
| `run_experiment.py` | Training and five-mode generation evaluation |
| `run_paper_protocol.py` | Config-driven paper runs and post-run validation |
| `analyze_cross_rule_watermarkability.py` | Shared-trajectory watermarkability analysis |
| `analyze_composition_pool_paired.py` | Paired PPL reductions and bootstrap intervals |
| `paper_results/data/` | Compact aggregate results used by the paper |

## Installation

Python 3.10 or newer is recommended. Paper-scale training and generation require
a CUDA-capable GPU.

```bash
python -m venv .venv
# Linux or macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify a fresh checkout without downloading a model or running a GPU job:

```bash
python -m unittest discover -p "test_*.py" -v
python paper_results/scripts/validate_results.py
python run_paper_protocol.py --config configs/opt_composition_pool.json --dry-run
```

The first command exercises the implementation. The second independently
recomputes key statistics from the released aggregate tables. The dry run shows
the exact training, run-validation, and paired-analysis commands that the
protocol launcher would execute.

## Quick Demo

The demo downloads `facebook/opt-125m` on first use, loads the released main-run
adapter and fixed partition, and compares the base model, the analytic
controller alone, and the adapter followed by the same controller:

```bash
python demo_watermarkability.py \
  --prompt "Photosynthesis is the process by which plants" \
  --rule binary \
  --tau 0.99 \
  --max-new-tokens 16
```

It reports the target-set mass before and after control, the residual logit bias
needed to reach the requested mass, sampled-token adherence, base-model PPL, and
the generated continuation. The default `tau=0.99`, temperature, adapter, and
partition match the paper setup. Changing a demo option illustrates the method
but does not reproduce a reported paper result. Available static schedules are
`binary`, `unigram`, `cw_abcd`, and `ecc`; the ECC schedule requires a token
count divisible by eight.

Use `--device cpu`, `--device cuda`, or the default `--device auto`. Add
`--local-files-only` after the base model has been cached. To verify the released
adapter and partition without loading a model, run:

```bash
python demo_watermarkability.py --check-artifacts
```

## Data and Partitions

Build the deterministic 5,000/500/256 LFQA prompt manifest from the public LFQA
train split:

```bash
python build_lfqa_manifest.py \
  --output data/lfqa_prompts_5756.jsonl \
  --train-count 5000 \
  --validation-count 500 \
  --test-count 256 \
  --seed 20260810
```

The builder uses cached Arrow shards when available and otherwise downloads
`vblagoje/lfqa`. It rejects empty prompts, unknown splits, and duplicate prompts
across splits.

The versioned `assets/atomic16/` directories are the exact fixed partitions used
by the saved paper runs. The directory name and internal code retain `atomic16`
for result compatibility; conceptually, these are the paper's 16 base buckets.
To rebuild a partition instead of using the released artifact:

```bash
python build_atomic16_partition.py \
  --model-name facebook/opt-125m \
  --manifest data/lfqa_prompts_5756.jsonl \
  --output-dir assets/atomic16/opt-125m-lfqa5000-v1 \
  --device cuda
```

The builder excludes tokenizer special IDs and output-matrix padding rows. It
does not require artifacts from another repository.

## Reproducing Paper Runs

The canonical composition-pool protocol trains on all 2-, 4-, 7-, and 8-of-16
base-bucket unions without exposing an evaluated watermark identity. With 5,000
prompts repeated four times, it uses 20,000 training and 2,000 validation
trajectories. Watermark schedules are attached only on the 256 held-out prompts.

| Config | Paper role |
|---|---|
| `configs/opt_composition_pool.json` | Main OPT composition-pool result |
| `configs/qwen3_composition_pool.json` | Main Qwen3 composition-pool result |
| `configs/opt_schedule_aware.json` | OPT schedule-aware appendix control |
| `configs/qwen3_schedule_aware.json` | Qwen3 schedule-aware appendix control |

The canonical configs set `local_files_only: true` so a scheduled run cannot
silently download a different model artifact. Before launching one, cache the
specified Hugging Face model and tokenizer locally. For an initial download,
use a copied config with `local_files_only` set to `false`, then restore the
canonical setting for the recorded run.

Run a protocol locally or inside an allocated GPU job:

```bash
python run_paper_protocol.py --config configs/opt_composition_pool.json
```

The launcher refuses to reuse an existing output directory unless
`--allow-existing-output` is supplied intentionally. For Slurm, an optional
scheduler-only wrapper is provided:

```bash
export WM_LORA_ROOT="$HOME/LLM_Watermarkability"
sbatch scripts/hpc/run_config.slurm configs/opt_composition_pool.json
```

Cluster resources may need adjustment. Scientific settings remain in the JSON
config rather than the scheduler wrapper.

## Validation and Artifacts

Validate a completed OPT main run with:

```bash
python validate_atomic16_pipeline.py \
  --partition-dir assets/atomic16/opt-125m-lfqa5000-v1 \
  --experiment-dir outputs/opt125m_composition_pool_release_lfqa5000x4_rank16_epochs6_seed20260810 \
  --expected-rules ecc,binary,cw_abcd,unigram,atomic_kgw,atomic_kgw25,entropy_kgw
```

Trajectory caches include hashes of the manifest, ordered prompts, target
arrays, partition files, tokenizer, model, and generation settings. Stale
caches are rejected rather than silently reused. Adapter checkpoints require an
exact trainable-key match when loaded.

Compact aggregate tables and their self-contained validator are released under
`paper_results/data/` and `paper_results/scripts/validate_results.py`. Model
checkpoints other than the small OPT adapter used by the demo, prompt manifests,
token-level generation archives, manuscript sources, and local review files are
intentionally excluded. See `IMPLEMENTATION_AUDIT.md` for the implementation
audit.

## Citation

```bibtex
@article{deng2026watermarkability,
  title   = {Watermarkability: Measuring and Improving Language-Model Compatibility with Logit-Based Watermarks},
  author  = {Deng, Zewei and Seif, Mohamed and Xie, Liyan},
  journal = {arXiv preprint arXiv:2610.22810},
  year    = {2026}
}
```

## License

The original code and aggregate result tables in this repository are released
under the MIT License. The released OPT-125M adapter is derived from
`facebook/opt-125m`, is not covered by the repository's MIT License, and is
provided for non-commercial research under the OPT-175B License. See
`checkpoints/opt125m_composition_pool/NOTICE.md` and `OPT_LICENSE.md`. Models and
other third-party dependencies downloaded separately retain their respective
licenses.
