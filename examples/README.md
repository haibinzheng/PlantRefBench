# Synthetic schema example

`synthetic_orthogroup_cells.csv` is a tiny, fully invented cell-by-orthogroup table. Its five development species and two held-out target species mirror the study design, but **no row or value comes from the study datasets**.

Target labels are absent from the feature table. `synthetic_target_truth.csv` is a separate toy-only file illustrating that evaluation labels should be opened only after predictions have been saved.

This example documents a data shape; the archived research runners use H5AD inputs and additional mappings, so this CSV is not a substitute for the experimental datasets or a basis for reproducing paper metrics.
