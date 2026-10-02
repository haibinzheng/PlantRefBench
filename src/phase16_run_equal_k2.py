#!/usr/bin/env python3
"""Generate label-blind Phase 16 predictions for one frozen target and seed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.linear_model import SGDClassifier


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
import sys
sys.path.insert(0, str(PROJECT / "src"))
import run_phase2_orthogroup_baseline as ogbase  # noqa: E402

CONFIG = PROJECT / "configs/phase16_reference_mechanism_v1.json"
MANIFEST = PROJECT / "configs/pilot_manifest.csv"
LABEL_MAPPING = PROJECT / "configs/root_label_mapping_draft.csv"
REPORT_ROOT = Path("/data/reports/phylo_plant_fm_pilot/phase16_reference_mechanism_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def load_feature_cache(cfg: dict) -> dict[str, dict]:
    root = Path(cfg["feature_cache"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("labels_read") is not False:
        raise RuntimeError("Phase 16 feature cache is incomplete or is not label-free")
    if manifest.get("config_sha256") != sha256(CONFIG):
        raise RuntimeError("Feature cache belongs to a different frozen configuration")
    data = {}
    for item in manifest["entries"]:
        matrix_path, obs_path = Path(item["matrix_file"]), Path(item["obs_file"])
        if sha256(matrix_path) != item["matrix_sha256"] or sha256(obs_path) != item["obs_sha256"]:
            raise RuntimeError(f"Feature cache hash mismatch: {item['species']}")
        data[item["species"]] = {
            "x": sparse.load_npz(matrix_path).tocsr(),
            "obs": pd.read_csv(obs_path)["obs_name"].astype(str).to_numpy(),
            "kind": item["kind"],
        }
    return data


def load_phase12_sources(cfg: dict, excluded_target: str, seed: int, data: dict[str, dict]) -> tuple[dict[str, sparse.csr_matrix], dict[str, np.ndarray]]:
    """Load only source labels and reproduce frozen Phase 12 source sampling.

    The held-out development target is excluded before opening any obs column;
    its labels, like external target labels, are opened only by aggregation.
    """
    rows = [row for row in csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")) if row["role"] == "train"]
    mapping = pd.read_csv(LABEL_MAPPING).fillna("")
    label_map = dict(mapping.loc[mapping["mapping_status"].eq("mapped"), ["source_label", "canonical_family"]].itertuples(index=False, name=None))
    order = {row["dataset_id"]: index for index, row in enumerate(rows)}
    matrices, labels = {}, {}
    for row in rows:
        if row["species"] == excluded_target:
            continue
        adata = ad.read_h5ad(row["source_h5ad"], backed="r")
        try:
            canonical = adata.obs[ogbase.source_label_column(adata)].astype(str).map(label_map)
            positions = np.flatnonzero((canonical.notna() & canonical.ne("")).to_numpy())
            local_labels = canonical.iloc[positions].reset_index(drop=True)
            local = ogbase.stratified_indices(local_labels, int(row["max_cells"]), int(seed) + 1009 * order[row["dataset_id"]])
            selected = positions[local]
            selected_labels = local_labels.iloc[local].to_numpy(dtype=str)
        finally:
            adata.file.close()
        if selected.max(initial=-1) >= data[row["species"]]["x"].shape[0]:
            raise RuntimeError(f"Feature cache row mismatch for {row['species']}")
        matrices[row["species"]] = data[row["species"]]["x"][selected]
        labels[row["species"]] = selected_labels
    return matrices, labels


def source_combinations(target: str, cfg: dict) -> list[tuple[str, str]]:
    candidates = [s for s in cfg["source_order"] if s != target]
    return list(itertools.combinations(candidates, int(cfg["reference_species_per_combination"])))


def pair_distance(a: str, b: str, cfg: dict) -> int:
    table = cfg["source_pairwise_distance_ordinal"]
    return int(table.get(a, {}).get(b, table.get(b, {}).get(a)))


def design_row(target: str, combo: tuple[str, str], labels: dict[str, np.ndarray], common: list[str], cfg: dict) -> dict:
    common_set = set(common)
    counts = {s: pd.Series(labels[s]).value_counts() for s in combo}
    supports = [sum(int(counts[s].get(family, 0) > 0) for s in combo) for family in common]
    total_family_cells = [sum(int(counts[s].get(family, 0)) for s in combo) for family in common]
    richness = len(set.union(*(set(labels[s]) for s in combo)))
    distances = [int(cfg["target_distance_ordinal"][target][s]) for s in combo]
    return {
        "target_species": target,
        "combination_id": "__".join(s.lower().replace(" ", "_") for s in combo),
        "source_species": json.dumps(combo),
        "source_species_count": len(combo),
        "reference_pairwise_breadth": pair_distance(combo[0], combo[1], cfg),
        "mean_target_distance": float(np.mean(distances)),
        "minimum_target_distance": int(min(distances)),
        "maximum_target_distance": int(max(distances)),
        "family_richness_before_common_class_filter": richness,
        "mean_family_species_support": float(np.mean(supports)),
        "fraction_families_supported_by_all_references": float(np.mean(np.asarray(supports) == len(combo))),
        "mean_combined_source_cells_per_common_family": float(np.mean(total_family_cells)),
        "minimum_combined_source_cells_per_common_family": int(min(total_family_cells)),
    }


def classifier(seed: int, cfg: dict) -> SGDClassifier:
    settings = cfg["classifier"]
    return SGDClassifier(
        loss=settings["loss"], alpha=float(settings["alpha"]), class_weight=settings["class_weight"],
        max_iter=int(settings["max_iter"]), tol=float(settings["tol"]), random_state=seed, n_jobs=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    allowed = set(cfg["development_species"] + cfg["external_species"])
    if args.target not in allowed or args.seed not in cfg["seeds"]:
        raise ValueError("Target or seed is outside the frozen Phase 16 protocol")
    data = load_feature_cache(cfg)
    source_matrices, labels = load_phase12_sources(cfg, args.target, args.seed, data)
    combinations = source_combinations(args.target, cfg)
    if not combinations:
        raise RuntimeError("No valid frozen combinations")
    # The intersection is calculated from sources alone over *all* declared pairs.
    pair_vocabularies = [set(np.concatenate([labels[s] for s in pair])) for pair in combinations]
    common = sorted(set.intersection(*pair_vocabularies))
    if not common:
        raise RuntimeError("No common source class across the predeclared combinations")
    common_set = set(common)
    candidates = sorted(set(itertools.chain.from_iterable(combinations)), key=cfg["source_order"].index)
    filtered = {}
    for species in candidates:
        keep = np.isin(labels[species], common)
        filtered[species] = {"x": source_matrices[species][keep], "y": labels[species][keep]}
    quota = min(len(filtered[species]["y"]) for species in candidates)
    if quota < 2:
        raise RuntimeError(f"Unusable equal-per-source quota: {quota}")
    design = [design_row(args.target, pair, labels, common, cfg) for pair in combinations]
    target_x, target_obs = data[args.target]["x"], data[args.target]["obs"]
    output = REPORT_ROOT / "runs" / f"{args.target.lower().replace(' ', '_')}_seed{args.seed}"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite completed Phase 16 run: {output}")
    output.mkdir(parents=True)
    prediction_frames = []
    for pair, row in zip(combinations, design):
        xs, ys = [], []
        for species in pair:
            rng = np.random.default_rng(stable_seed(cfg["run_id"], args.target, args.seed, species))
            chosen = np.sort(rng.choice(len(filtered[species]["y"]), quota, replace=False))
            xs.append(filtered[species]["x"][chosen])
            ys.append(filtered[species]["y"][chosen])
        x_train, y_train = sparse.vstack(xs, format="csr"), np.concatenate(ys)
        model = classifier(stable_seed(cfg["run_id"], args.target, args.seed, "model"), cfg)
        model.fit(x_train, y_train)
        prediction_frames.append(pd.DataFrame({
            "target_species": args.target, "seed": args.seed, "combination_id": row["combination_id"],
            "obs_name": target_obs, "prediction": model.predict(target_x),
            "common_classes": json.dumps(common), "per_source_quota": quota,
            "matched_train_cells": len(y_train),
        }))
    prediction_path = output / "predictions_blinded.csv.gz"
    pd.concat(prediction_frames, ignore_index=True).to_csv(prediction_path, index=False, compression="gzip")
    design_frame = pd.DataFrame(design)
    design_frame["seed"] = args.seed
    design_frame["per_source_quota"] = quota
    design_frame["matched_train_cells"] = quota * int(cfg["reference_species_per_combination"])
    design_frame["common_train_classes"] = len(common)
    design_frame.to_csv(output / "combination_design.csv", index=False)
    summary = {
        "run_id": cfg["run_id"], "status": "completed", "target_species": args.target, "seed": args.seed,
        "target_kind": data[args.target]["kind"], "combination_count": len(combinations),
        "common_train_classes": len(common), "per_source_quota": quota,
        "matched_train_cells": quota * int(cfg["reference_species_per_combination"]),
        "target_labels_read": False, "predictions_saved_before_label_open": True,
        "blinded_predictions_file": str(prediction_path), "blinded_predictions_sha256": sha256(prediction_path),
        "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
