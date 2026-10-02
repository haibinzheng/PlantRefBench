# Running and input requirements

## Offline smoke checks

From the repository root with Python 3.11 or newer:

```sh
python scripts/check_package.py
python examples/validate_example.py
python -m unittest discover -s tests -v
```

These commands use only the Python standard library. They parse archived source without importing it, validate configuration syntax and named local dependencies, and check the invented example's IDs, nonnegative finite features and separation of target labels. They do not train a classifier, validate scientific performance, or reproduce the study. The example validator deliberately reads toy truth to check alignment; it is not the label-blind prediction stage.

## Research environments

Use an isolated environment for research runners. `python -m pip install -r requirements-core.txt` installs the common preparation and linear-analysis dependencies. This is a dependency list, **not** an exact historical environment lock and not a claim that every current dependency combination has been tested.

scANVI additionally requires compatible PyTorch, scvi-tools and Scanpy installations; SAMap requires its own compatible SAMap/Scanpy stack and reciprocal sequence mappings. Select CUDA/PyTorch versions for the available hardware. Do not combine optional model environments blindly. `src/phase17_install_samap_overlay_offline.sh` is an archived server-specific offline installer requiring an existing base environment and wheelhouse, not a general installation command.

## Input contracts

| Input | Purpose | Included? |
| --- | --- | --- |
| `configs/pilot_manifest.csv` | Dataset IDs, roles, historical H5AD paths and caps | Yes, metadata only |
| `configs/root_label_mapping_draft.csv` | Historical source-label to cell-family rules | Yes; original filename retained |
| Source H5AD files | Expression and observation metadata | No |
| Gene-ID mapping and OrthoFinder orthogroup tables | Shared orthogroup representation | No |
| Catharanthus-to-frozen-orthogroup mapping | External target representation | Preparation code included; generated mapping absent |
| Reciprocal sequence maps | SAMap cross-species mapping | No; reuse existing maps when continuing a frozen run |
| Cached counts/features and seed-specific manifests | Prepared model inputs | No; preparation code included |
| Prior predictions, hashes and summaries | Aggregation/stability reference inputs | No |

The pilot manifest also records early study-holdout datasets. Its row count must not be interpreted as the number of targets in every later experiment. Later phase JSON configurations define their own scope. Preserve the original data's usage terms and citations when obtaining inputs; synthetic examples grant no rights to source datasets.

## Archived workflow entry points

| Stage | Relevant source files |
| --- | --- |
| Shared-orthogroup linear baseline | `run_phase2_study_holdout_baseline.py`, `run_phase2_orthogroup_baseline.py`, `run_phase2_loso_shared_orthogroup.py` |
| Equal-reference-budget mechanism | `phase16_prepare_feature_cache.py`, `phase16_run_equal_k2.py`, `phase16_aggregate_mechanism.py` |
| External linear validation | `prepare_catharanthus_frozen_mapping.py`, `run_external_linear_validation_v1.py`, `aggregate_external_validation_v1.py` |
| External scANVI | `build_scvi_orthogroup_counts.py`, `prepare_external_scanvi_counts_v1.py`, `run_external_scanvi_validation_v1.py`, `aggregate_external_scanvi_validation_v1.py` |
| External SAMap | `phase17_prepare_external_inputs_v1.py`, `phase17_build_external_maps_v1.sh`, `phase17_run_external_condition_v1.py`, `phase17_aggregate_external_v1.py` |
| Stability | `phase18b_*` preparation, condition and aggregation scripts |
| Linear budget dose | `phase18c_run_linear_budget_dose_v1.py`, `phase18c_aggregate_linear_budget_dose_v1.py`, `phase18c_dose_statistics_v1.py` |
| Family and coverage evaluation | `phase18d_*`, `phase18f_build_supplementary_evaluation_v1.py` |

This is a dependency guide, not a command to run every archived stage. Inspect each runner's arguments, configuration and input-path constants before execution. Several runners have hard-coded `/workspace/projects/phylo_plant_fm_pilot` and `/data/.../phylo_plant_fm_pilot` paths, and aggregators require earlier outputs. Adapt a separate working copy to your installation and record the changes; do not overwrite an existing frozen experiment. In particular, never rerun map construction or finished predictions merely to satisfy packaging checks.

Save and hash label-blind predictions before evaluation, retain target IDs, and keep paired seeds/budgets aligned. Random seeds are technical repeats, not biological replicates. No full-data rerun was performed for this packaging update.

