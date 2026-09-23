#!/usr/bin/env python3
"""Run one frozen, label-blind external scANVI reference-composition condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy import sparse


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
TARGET_CACHE_ROOT = Path("/data/derived/phylo_plant_fm_pilot/external_validation_v1/external_scanvi_counts_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scvi.settings.seed = seed


def load_development(cache: Path):
    manifest = json.loads((cache / "manifest.json").read_text())
    vocabulary = np.asarray((cache / "orthogroup_vocabulary.txt").read_text().splitlines())
    data = {}
    for item in manifest["datasets"]:
        matrix_path, obs_path = Path(item["matrix_file"]), Path(item["obs_file"])
        if sha256(matrix_path) != item["matrix_sha256"] or sha256(obs_path) != item["obs_sha256"]:
            raise RuntimeError(f"Development cache hash mismatch: {item['dataset_id']}")
        data[item["species"]] = {
            "x": sparse.load_npz(matrix_path).tocsr(),
            "obs": pd.read_csv(obs_path),
            "dataset_id": item["dataset_id"],
        }
    return manifest, vocabulary, data


def hvg_mask(training: sparse.csr_matrix, vocabulary: np.ndarray, count: int):
    temporary = ad.AnnData(X=training.copy(), var=pd.DataFrame(index=vocabulary))
    sc.pp.normalize_total(temporary, target_sum=1e4)
    sc.pp.log1p(temporary)
    sc.pp.highly_variable_genes(temporary, flavor="seurat", n_top_genes=count, subset=False)
    mask = temporary.var["highly_variable"].to_numpy(dtype=bool)
    if mask.sum() != min(count, training.shape[1]):
        raise RuntimeError(f"Unexpected HVG count: {mask.sum()}")
    return mask


def history_summary(model):
    return {key: [float(x) for x in np.asarray(values).reshape(-1)] for key, values in model.history.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", required=True, choices=["Sorghum bicolor", "Catharanthus roseus"])
    parser.add_argument("--condition", required=True, choices=["all_sources", "nearest_clade"])
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    if args.seed not in cfg["seeds"] or args.seed == cfg["reused_seed"]:
        raise RuntimeError("Seed is not an authorized new Phase 18B seed")
    report_root = Path(cfg["report_root"])
    checkpoint_root = Path(cfg["checkpoint_root"])
    target_cfg = cfg["reference_conditions"][args.target]
    condition_offset = 0 if args.condition == "all_sources" else 101
    model_seed = int(args.seed) + 1000 * int(target_cfg["target_index"]) + condition_offset
    set_seed(model_seed)

    cache = Path(cfg["development_cache"])
    dev_manifest, vocabulary, data = load_development(cache)
    condition_sources = {
        "all_sources": target_cfg["all_sources"],
        "nearest_clade": target_cfg["nearest_clade"],
    }
    label_sets = {}
    for name, species_list in condition_sources.items():
        label_sets[name] = set(np.concatenate([data[species]["obs"]["canonical_family"].to_numpy(str) for species in species_list]))
    common_classes = sorted(set.intersection(*label_sets.values()))
    common_set = set(common_classes)
    pools = {}
    for name, species_list in condition_sources.items():
        xs, ys, source_rows = [], [], []
        for species in species_list:
            labels = data[species]["obs"]["canonical_family"].to_numpy(str)
            keep = np.asarray([label in common_set for label in labels])
            xs.append(data[species]["x"][keep])
            ys.append(labels[keep])
            source_rows.extend([species] * int(keep.sum()))
        pools[name] = {
            "x": sparse.vstack(xs, format="csr"),
            "y": np.concatenate(ys),
            "source_species": np.asarray(source_rows, dtype=object),
        }
    matched_budget = min(len(pool["y"]) for pool in pools.values())
    pool = pools[args.condition]
    rng = np.random.default_rng(model_seed + 4242)
    chosen = np.sort(rng.choice(len(pool["y"]), matched_budget, replace=False))
    x_train_full = pool["x"][chosen]
    y_train = pool["y"][chosen]
    sampled_sources = pool["source_species"][chosen]
    hvg = hvg_mask(x_train_full, vocabulary, int(cfg["hvg_count"]))
    selected_vocabulary = vocabulary[hvg]
    x_train = x_train_full[:, hvg].astype(np.float32)

    target_slug = args.target.lower().replace(" ", "_")
    target_cache = TARGET_CACHE_ROOT / target_slug
    target_summary = json.loads((target_cache / "summary.json").read_text())
    target_matrix_path, target_obs_path = Path(target_summary["matrix_file"]), Path(target_summary["obs_file"])
    if sha256(target_matrix_path) != target_summary["matrix_sha256"] or sha256(target_obs_path) != target_summary["obs_sha256"]:
        raise RuntimeError("External target cache hash mismatch")
    x_target = sparse.load_npz(target_matrix_path).tocsr()[:, hvg].astype(np.float32)
    target_obs = pd.read_csv(target_obs_path)["obs_name"].astype(str).to_numpy()

    categories = common_classes + ["Unknown"]
    train_obs = pd.DataFrame(
        {"label": pd.Categorical(y_train, categories=categories)},
        index=[f"train_{i}" for i in range(len(y_train))],
    )
    query_obs = pd.DataFrame(
        {"label": pd.Categorical(["Unknown"] * len(target_obs), categories=categories)},
        index=[f"query_{i}" for i in range(len(target_obs))],
    )
    train_adata = ad.AnnData(X=x_train, obs=train_obs, var=pd.DataFrame(index=selected_vocabulary))
    query_adata = ad.AnnData(X=x_target, obs=query_obs, var=pd.DataFrame(index=selected_vocabulary))
    scvi.model.SCVI.setup_anndata(train_adata)
    vae = scvi.model.SCVI(
        train_adata,
        n_hidden=int(cfg["n_hidden"]),
        n_latent=int(cfg["n_latent"]),
        n_layers=int(cfg["n_layers"]),
        dropout_rate=float(cfg["dropout_rate"]),
        dispersion="gene",
        gene_likelihood="nb",
    )
    start = time.time()
    vae.train(
        max_epochs=int(cfg["scvi_max_epochs"]), train_size=0.9, validation_size=0.1,
        batch_size=int(cfg["batch_size"]), accelerator="gpu", devices=1,
        early_stopping=True, early_stopping_patience=5, check_val_every_n_epoch=1,
        enable_progress_bar=False,
    )
    scvi_seconds = time.time() - start
    scanvi_model = scvi.model.SCANVI.from_scvi_model(vae, labels_key="label", unlabeled_category="Unknown")
    start = time.time()
    scanvi_model.train(
        max_epochs=int(cfg["scanvi_max_epochs"]), train_size=0.9, validation_size=0.1,
        batch_size=int(cfg["batch_size"]), accelerator="gpu", devices=1,
        early_stopping=True, early_stopping_patience=5, check_val_every_n_epoch=1,
        enable_progress_bar=False,
    )
    scanvi_seconds = time.time() - start
    prediction = np.asarray(scanvi_model.predict(query_adata), dtype=str)

    run_name = f"{target_slug}_{args.condition}_seed{args.seed}"
    report_dir = report_root / run_name
    checkpoint_dir = checkpoint_root / run_name
    report_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = report_dir / "predictions_blinded.csv.gz"
    pd.DataFrame({
        "obs_name": target_obs,
        "prediction": prediction,
        "common_classes": json.dumps(common_classes),
        "matched_train_cells": matched_budget,
    }).to_csv(prediction_path, index=False)
    prediction_sha = sha256(prediction_path)
    scanvi_model.save(checkpoint_dir / "scanvi", overwrite=False, save_anndata=False)
    pd.DataFrame({"orthogroup": selected_vocabulary}).to_csv(report_dir / "selected_hvgs.csv", index=False)
    (report_dir / "training_history.json").write_text(json.dumps(
        {"scvi": history_summary(vae), "scanvi": history_summary(scanvi_model)}, indent=2
    ) + "\n")
    summary = {
        "run_id": cfg["run_id"],
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "target_species": args.target,
        "condition": args.condition,
        "seed": int(args.seed),
        "model_seed": model_seed,
        "source_species": sorted(set(sampled_sources)),
        "matched_train_cells": int(len(y_train)),
        "common_train_classes": len(common_classes),
        "target_cells_predicted": len(target_obs),
        "hvg_count": int(hvg.sum()),
        "scvi_seconds": scvi_seconds,
        "scanvi_seconds": scanvi_seconds,
        "target_expression_used_in_fit": False,
        "target_labels_read": False,
        "predictions_saved_before_label_open": True,
        "blinded_predictions_file": str(prediction_path),
        "blinded_predictions_sha256": prediction_sha,
        "checkpoint_dir": str(checkpoint_dir),
        "versions": {"scvi": scvi.__version__, "torch": torch.__version__, "cuda": torch.version.cuda},
        "input_sha256": {
            "config": sha256(config_path),
            "development_manifest": sha256(cache / "manifest.json"),
            "target_summary": sha256(target_cache / "summary.json"),
            "code": sha256(Path(__file__)),
        },
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
