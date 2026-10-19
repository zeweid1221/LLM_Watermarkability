# Validation Checklist

Run from the repository root after installing `requirements.txt`.

```bash
python -m compileall -q .
python -m unittest discover -p "test_*.py" -v
```

The suite checks the controller equations, LoRA gradients, static and dynamic
target masks, exhaustive composition coverage, prompt-tokenization consistency,
cache fingerprints, partition identity checks, adapter checkpoint integrity,
cross-rule formulas, teacher-forced watermarkability statistics, and exact
materialization of the three released paper protocol configs.

For each completed model run, also execute `validate_atomic16_pipeline.py` with
the corresponding fixed partition and expected rule list. Run
`validate_cross_rule_watermarkability.py` on both model directories before
combining the cross-rule results.

The paper source has a separate numeric consistency check:

```bash
python paper_results/scripts/validate_results.py
```

Small smoke runs verify plumbing only and must not be reported as scientific
results. See `IMPLEMENTATION_AUDIT.md` for release findings that affect result
compatibility.
