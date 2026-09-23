#!/usr/bin/env python3
"""Run one frozen SAMap external condition and graph-based label transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, f1_score

PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
sys.path.insert(0, str(PROJECT / "src"))
import run_phase2_orthogroup_baseline as ogbase  # noqa: E402

CONFIG = PROJECT / "configs/phase17_samap_external_v1.json"
MANIFEST = PROJECT / "configs/pilot_manifest.csv"
LABEL_MAPPING = PROJECT / "configs/root_label_mapping_draft.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_row(species: str, role: str) -> dict[str, str]:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    return next(row for row in rows if row["species"] == species and row["role"] == role)


def canonical_lookup() -> dict[str, str]:
    mapping = pd.read_csv(LABEL_MAPPING)
    mapped = mapping.loc[mapping["mapping_status"].eq("mapped"), ["source_label", "canonical_family"]]
    return dict(mapped.itertuples(index=False, name=None))


def raw_label_series(code: str, cfg: dict, lookup: dict[str, str]) -> pd.Series:
    spec = cfg["species"][code]
    row = manifest_row(spec["species"], spec["role"])
    raw = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        column = ogbase.source_label_column(raw)
        values = raw.obs[column].astype(str).map(lookup).fillna("__unmapped__").to_numpy()
        return pd.Series(values, index=pd.Index(raw.obs_names.astype(str)))
    finally:
        raw.file.close()


def resolve_deduplicated_names(names: pd.Index, reference: pd.Index,
                               context: str) -> tuple[pd.Index, int]:
    resolved = names.astype(str).to_list()
    normalized = 0
    for i, name in enumerate(names.astype(str)):
        if name in reference:
            continue
        candidate = re.sub(r"-\d+$", "", name)
        if candidate == name or candidate not in reference:
            raise RuntimeError(f"Cannot align {context} obs_names to frozen labels")
        resolved[i] = candidate
        normalized += 1
    result = pd.Index(resolved)
    if not result.is_unique:
        raise RuntimeError(f"{context} obs_name normalization is not one-to-one")
    if normalized:
        print(f"INFO: Normalized {normalized} deterministic deduplication suffix(es) for {context}")
    return result, normalized


def labels_for_stitched(names: pd.Index, code: str, source: pd.Series) -> np.ndarray:
    stripped = names.str.removeprefix(f"{code}_")
    try:
        lookup_names, _ = resolve_deduplicated_names(stripped, source.index, f"stitched {code}")
    except RuntimeError:
        lookup_names, _ = resolve_deduplicated_names(names, source.index, f"stitched {code}")
    values = source.reindex(lookup_names)
    if values.isna().any():
        raise RuntimeError(f"Missing stitched labels for {code}")
    return values.astype(str).to_numpy()


def install_stringarray_shim() -> bool:
    cls = pd.arrays.StringArray
    if hasattr(cls, "flatten"):
        return False
    cls.flatten = lambda self: self.to_numpy().flatten()
    return True


def predict_from_graph(stitched: ad.AnnData, target_code: str, sources: list[str],
                       vocabulary: list[str], cfg: dict) -> pd.DataFrame:
    species = stitched.obs["species"].astype(str).to_numpy()
    names = pd.Index(stitched.obs_names.astype(str))
    label_to_index = {label: i for i, label in enumerate(vocabulary)}
    node_labels = np.full(stitched.n_obs, -1, dtype=np.int32)
    lookup = canonical_lookup()
    for code in sources:
        positions = np.flatnonzero(species == code)
        values = labels_for_stitched(names[positions], code, raw_label_series(code, cfg, lookup))
        node_labels[positions] = np.asarray([label_to_index.get(value, -1) for value in values], dtype=np.int32)
    target_positions = np.flatnonzero(species == target_code)
    graph = sparse.csr_matrix(stitched.obsp["connectivities"])
    predictions, weights, counts = [], [], []
    for position in target_positions:
        row = graph.getrow(position)
        labels = node_labels[row.indices]
        keep = labels >= 0
        if not np.any(keep):
            predictions.append("__unassigned__")
            weights.append(0.0)
            counts.append(0)
            continue
        totals = np.bincount(labels[keep], weights=row.data[keep], minlength=len(vocabulary))
        winner = int(np.flatnonzero(totals == totals.max())[0])
        predictions.append(vocabulary[winner])
        weights.append(float(totals[winner]))
        counts.append(int(np.sum(keep)))
    original_names = names[target_positions].str.removeprefix(f"{target_code}_")
    return pd.DataFrame({"target_obs_name": original_names, "prediction": predictions,
                         "winning_weight": weights, "source_neighbor_count": counts})


def score_predictions(predictions: pd.DataFrame, target_code: str, vocabulary: list[str],
                      cfg: dict) -> dict:
    truth = raw_label_series(target_code, cfg, canonical_lookup())
    requested = pd.Index(predictions["target_obs_name"].astype(str))
    resolved_index, normalized = resolve_deduplicated_names(
        requested, truth.index, "target prediction"
    )
    aligned = truth.reindex(resolved_index)
    if aligned.isna().any():
        raise RuntimeError("Target prediction rows do not align to frozen labels")
    mask = aligned.isin(vocabulary).to_numpy()
    y_true = aligned.to_numpy()[mask]
    y_pred = predictions["prediction"].to_numpy()[mask]
    labels = sorted(set(y_true))
    family_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "evaluated_cells": int(mask.sum()), "total_target_cells": int(len(predictions)),
        "common_source_vocabulary": vocabulary,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "prediction_coverage": float(np.mean(predictions["prediction"].to_numpy() != "__unassigned__")),
        "unassigned_cells": int(np.sum(predictions["prediction"].to_numpy() == "__unassigned__")),
        "obs_name_normalized_cells": int(normalized),
        "per_family_f1": {label: float(value) for label, value in zip(labels, family_f1)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["sb", "cr"], required=True)
    parser.add_argument("--condition", choices=["all_sources", "nearest_clade"], required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-scoring-only", action="store_true")
    resume.add_argument("--resume-from-blind", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if importlib.metadata.version("sc-samap") != "3.0.0":
        raise RuntimeError("Unexpected SAMap version")
    input_manifest_path = Path(cfg["prepared_root"]) / "inputs" / "manifest.json"
    map_manifest_path = Path(cfg["prepared_root"]) / "maps_manifest.json"
    inputs = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    maps = json.loads(map_manifest_path.read_text(encoding="utf-8"))
    if maps["pair_count"] != 20:
        raise RuntimeError("Shared map manifest is incomplete")
    target_info = inputs["targets"][args.target]
    condition_info = target_info["conditions"][args.condition]
    sources = condition_info["sources"]
    codes = sources + [args.target]
    sams = {code: condition_info["inputs"][code]["prepared_h5ad"] for code in codes}
    report = Path(cfg["report_root"]) / args.target / args.condition
    if args.resume_from_blind:
        if not report.is_dir() or (report / "summary.json").exists():
            raise RuntimeError("Blind resume requires an incomplete existing report")
        if (report / "prediction_summary_blind.json").exists():
            raise RuntimeError("Blind resume refuses an existing prediction summary")
        blind = json.loads((report / "blind_summary.json").read_text(encoding="utf-8"))
        blind_path = Path(blind["stitched_h5ad"])
        if sha256(blind_path) != blind["stitched_h5ad_sha256"]:
            raise RuntimeError("Saved blind mapping hash does not match")
        stitched = ad.read_h5ad(blind_path)
        vocabulary = target_info["common_source_vocabulary"]
        predictions = predict_from_graph(stitched, args.target, sources, vocabulary, cfg)
        prediction_path = report / "predictions_blind.csv"
        predictions.to_csv(prediction_path, index=False)
        prediction_summary = {
            "target_labels_opened": False, "predictions": str(prediction_path),
            "predictions_sha256": sha256(prediction_path), "rows": int(len(predictions)),
            "resumed_from_frozen_blind_mapping": True,
        }
        (report / "prediction_summary_blind.json").write_text(
            json.dumps(prediction_summary, indent=2) + "\n", encoding="utf-8"
        )
        if sha256(prediction_path) != prediction_summary["predictions_sha256"]:
            raise RuntimeError("Prediction hash changed before target-label scoring")
        metrics = score_predictions(predictions, args.target, vocabulary, cfg)
        final = {
            "run_id": cfg["run_id"], "status": "completed", "target": args.target,
            "condition": args.condition, "blind_mapping": blind,
            "blind_predictions": prediction_summary, "metrics": metrics,
            "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)),
            "resumed_from_frozen_blind_mapping": True,
            "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
        (report / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(final, indent=2))
        return
    if args.resume_scoring_only:
        if not report.is_dir() or (report / "summary.json").exists():
            raise RuntimeError("Scoring-only resume requires an incomplete existing report")
        blind = json.loads((report / "blind_summary.json").read_text(encoding="utf-8"))
        prediction_summary = json.loads(
            (report / "prediction_summary_blind.json").read_text(encoding="utf-8")
        )
        prediction_path = Path(prediction_summary["predictions"])
        if sha256(prediction_path) != prediction_summary["predictions_sha256"]:
            raise RuntimeError("Saved blind prediction hash does not match")
        predictions = pd.read_csv(prediction_path)
        vocabulary = target_info["common_source_vocabulary"]
        metrics = score_predictions(predictions, args.target, vocabulary, cfg)
        final = {
            "run_id": cfg["run_id"], "status": "completed", "target": args.target,
            "condition": args.condition, "blind_mapping": blind,
            "blind_predictions": prediction_summary, "metrics": metrics,
            "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)),
            "scoring_resumed_from_frozen_blind_predictions": True,
            "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
        (report / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(final, indent=2))
        return
    if report.exists():
        raise RuntimeError(f"Refusing to overwrite condition report: {report}")
    report.mkdir(parents=True)
    shim = install_stringarray_shim()
    from samap import SAMAP
    sm = SAMAP(sams=sams, f_maps=str(Path(cfg["prepared_root"]) / "maps") + "/")
    sm.run()
    blind_path = report / "stitched_mapping_blind.h5ad"
    sm.samap.adata.write_h5ad(blind_path, compression="gzip")
    blind = {
        "saved_before_source_or_target_labels": True, "labels_in_samap_inputs": False,
        "stitched_h5ad": str(blind_path), "stitched_h5ad_sha256": sha256(blind_path),
        "samap_version": "3.0.0", "sources": sources, "target": args.target,
        "compatibility_shim_applied": shim,
    }
    (report / "blind_summary.json").write_text(json.dumps(blind, indent=2) + "\n", encoding="utf-8")
    vocabulary = target_info["common_source_vocabulary"]
    predictions = predict_from_graph(sm.samap.adata, args.target, sources, vocabulary, cfg)
    prediction_path = report / "predictions_blind.csv"
    predictions.to_csv(prediction_path, index=False)
    prediction_hash = sha256(prediction_path)
    prediction_summary = {
        "target_labels_opened": False, "predictions": str(prediction_path),
        "predictions_sha256": prediction_hash, "rows": int(len(predictions)),
    }
    (report / "prediction_summary_blind.json").write_text(
        json.dumps(prediction_summary, indent=2) + "\n", encoding="utf-8"
    )
    if sha256(prediction_path) != prediction_hash:
        raise RuntimeError("Prediction hash changed before target-label scoring")
    metrics = score_predictions(predictions, args.target, vocabulary, cfg)
    final = {
        "run_id": cfg["run_id"], "status": "completed", "target": args.target,
        "condition": args.condition, "blind_mapping": blind,
        "blind_predictions": prediction_summary, "metrics": metrics,
        "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)),
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (report / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
