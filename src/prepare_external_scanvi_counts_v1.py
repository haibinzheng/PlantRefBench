#!/usr/bin/env python3
"""Prepare label-blind external orthogroup count matrices for frozen scANVI inference."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
sys.path.insert(0, str(PROJECT / "src"))
import run_phase2_orthogroup_baseline as ogbase  # noqa: E402

MANIFEST = PROJECT / "configs/pilot_manifest.csv"
CONFIG = PROJECT / "configs/external_validation_v1/external_scanvi_validation_v1.json"
CATH_MAP = Path("/data/derived/phylo_plant_fm_pilot/external_validation_v1/catharanthus_mapping_v1/catharanthus_h5ad_to_frozen_orthogroup.tsv.gz")
OUT_ROOT = Path("/data/derived/phylo_plant_fm_pilot/external_validation_v1/external_scanvi_counts_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["Sorghum bicolor", "Catharanthus roseus"])
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    cache = Path(cfg["development_cache"])
    vocabulary = (cache / "orthogroup_vocabulary.txt").read_text().splitlines()
    feature_index = {og: index for index, og in enumerate(vocabulary)}
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    row = next(item for item in rows if item["species"] == args.target and item["role"] == "species_holdout")
    target_slug = args.target.lower().replace(" ", "_")
    out = OUT_ROOT / target_slug
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite external count cache: {out}")
    out.mkdir(parents=True)

    adata = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        var_to_feature = np.full(adata.n_vars, -1, dtype=np.int32)
        if args.target == "Catharanthus roseus":
            with gzip.open(CATH_MAP, "rt", encoding="utf-8", newline="") as handle:
                direct = {item["h5ad_gene"]: item["orthogroup"] for item in csv.DictReader(handle, delimiter="\t")}
            for position, h5_gene in enumerate(map(str, adata.var_names)):
                og = direct.get(h5_gene, "")
                if og in feature_index:
                    var_to_feature[position] = feature_index[og]
        else:
            training_keys = {ogbase.SPECIES_KEY[item["species"]] for item in rows if item["role"] == "train"}
            parsed_vocabulary, gene_to_og = ogbase.parse_orthogroups(training_keys)
            if parsed_vocabulary != vocabulary:
                raise RuntimeError("Frozen orthogroup vocabulary differs from parser output")
            safe_maps = ogbase.read_safe_gene_mappings()
            dataset_map = safe_maps[row["dataset_id"]]
            species_ogs = gene_to_og[ogbase.SPECIES_KEY[args.target]]
            for position, h5_gene in enumerate(map(str, adata.var_names)):
                reference_gene = dataset_map.get(h5_gene)
                og = species_ogs.get(reference_gene, "")
                if og in feature_index:
                    var_to_feature[position] = feature_index[og]
        gene_positions = np.flatnonzero(var_to_feature >= 0)
        matrix = sparse.csr_matrix(adata[:, gene_positions].to_memory().X)
        projection = sparse.csr_matrix(
            (
                np.ones(len(gene_positions), dtype=np.int64),
                (np.arange(len(gene_positions)), var_to_feature[gene_positions]),
            ),
            shape=(len(gene_positions), len(vocabulary)),
        )
        counts = (matrix @ projection).tocsr()
        counts.eliminate_zeros()
        if counts.data.size and ((counts.data < 0).any() or not np.equal(counts.data, np.rint(counts.data)).all()):
            raise RuntimeError("External matrix contains non-count values after orthogroup aggregation")
        counts.data = counts.data.astype(np.int64, copy=False)
        obs_names = np.asarray(adata.obs_names.astype(str))
    finally:
        adata.file.close()

    matrix_path = out / "target.counts.npz"
    obs_path = out / "target.obs_names.csv.gz"
    sparse.save_npz(matrix_path, counts, compressed=True)
    pd.DataFrame({"obs_name": obs_names}).to_csv(obs_path, index=False, compression="gzip")
    libraries = np.asarray(counts.sum(axis=1)).ravel()
    summary = {
        "run_id": "external_scanvi_counts_v1",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "target_species": args.target,
        "cells": int(counts.shape[0]),
        "features": int(counts.shape[1]),
        "nnz": int(counts.nnz),
        "mapped_input_genes": int(len(gene_positions)),
        "mean_active_features": float(counts.getnnz(axis=1).mean()),
        "library_min": int(libraries.min()),
        "library_median": float(np.median(libraries)),
        "library_max": int(libraries.max()),
        "labels_read": False,
        "matrix_file": str(matrix_path),
        "obs_file": str(obs_path),
        "matrix_sha256": sha256(matrix_path),
        "obs_sha256": sha256(obs_path),
        "config_sha256": sha256(CONFIG),
        "source_h5ad_sha256": sha256(Path(row["source_h5ad"])),
        "code_sha256": sha256(Path(__file__)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
