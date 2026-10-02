# PlantRefBench

Code and illustrative data for evaluating how reference-species composition affects cross-species plant single-cell cell-family transfer under matched reference-cell budgets.

## Scope

This repository contains analysis/model runners for orthogroup-based linear transfer, scANVI, and SAMap comparisons, plus a small **synthetic** schema example. The synthetic example is not a subset of the study data and must not be used to reproduce reported metrics.

The archived research scripts preserve the experiment's original path conventions and require separately obtained input datasets, orthogroup mappings, and configured Python environments. They are not a turnkey pipeline.

## Data and publication boundaries

- No original H5AD files, donor-level expression matrices, target labels, model checkpoints, or large derived data are included.
- Source-dataset redistribution rights and release-ready provenance will be checked before any real data are added.
- Manuscripts, drafts, submission files, and paper figures are intentionally excluded and will not be uploaded.
- The repository does not contain a new foundation model. It benchmarks reference-set design across existing model families.

## Layout

- `src/`: model and evaluation scripts.
- `configs/`: archived phase configurations, dataset inventory and cell-family mapping rules.
- `examples/`: explicitly synthetic input-schema examples.
- `scripts/` and `tests/`: offline package checks and regression tests.
- `docs/RUNNING.md`: environment guidance, missing external inputs and workflow entry points.
- `.gitignore`: guardrails against accidentally committing data, checkpoints, logs, reports, or manuscript files.

## Quick start: no GPU or study data required

```sh
python scripts/check_package.py
python examples/validate_example.py
python -m unittest discover -s tests -v
```

Use Python 3.11 or newer. These standard-library checks validate the package and synthetic input contract; they do not train models or reproduce reported performance. GitHub Actions runs the same checks on pushes and pull requests.

For research runners, see [Running and input requirements](docs/RUNNING.md). Common dependencies are listed in `requirements-core.txt`; optional scANVI/SAMap environments and required external mappings are separate. The archived scripts and configurations retain their original paths and parameters. Full study inputs, generated mappings, cached data and historical result artifacts are not bundled.

## License

No reuse license has been selected yet. Please contact the repository owner before redistributing or adapting the code.

