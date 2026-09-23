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
- `examples/`: explicitly synthetic input-schema examples.
- `.gitignore`: guardrails against accidentally committing data, checkpoints, logs, reports, or manuscript files.

## License

No reuse license has been selected yet. Please contact the repository owner before redistributing or adapting the code.
