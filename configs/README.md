# Paper Protocol Configs

These files contain the scientific settings used for the reported training and
evaluation runs. They are independent of Slurm and can be executed with:

```bash
python run_paper_protocol.py --config configs/CONFIG.json
```

Explicit `run_experiment.py` command-line arguments override config defaults for
small diagnostics. `run_paper_protocol.py` refuses to reuse an existing output
directory unless `--allow-existing-output` is supplied intentionally.

The two `composition_pool` configs define the main schedule-independent runs.
The two `schedule_aware` configs define the appendix controls trained on target
sets produced by the six primary schedule families plus the auxiliary
entropy-gated schedule. All four use the same 256 held-out LFQA prompts within
each model.

The canonical configs set `local_files_only` to `true`, so the corresponding
Hugging Face model and tokenizer must already be present in the local cache.
This avoids an implicit network dependency inside a scheduled paper run.
