#!/usr/bin/env python3
"""Build an exact Phase-12 rank-to-orthogroup cache without reading labels.

Each cell is represented by its top 256 raw input genes before mapping into
the frozen multicopy orthogroup vocabulary.  This matches Phase 12 and the
frozen external linear validation; it is not a count-aggregated sensitivity
representation.
"""

from __future__ import annotations

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

CONFIG = PROJECT / "configs/phase16_reference_mechanism_v1.json"
MANIFEST = PROJECT / "configs/pilot_manifest.csv"
CATH_MAP = Path("/data/derived/phylo_plant_fm_pilot/external_validation_v1/catharanthus_mapping_v1/catharanthus_h5ad_to_frozen_orthogroup.tsv.gz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_features(adata: ad.AnnData, row: dict, safe_maps: dict, gene_to_og: dict, index: dict[str, int]) -> np.ndarray:
    result = np.full(adata.n_vars, -1, dtype=np.int32)
    if row["species"] == "Catharanthus roseus":
        with gzip.open(CATH_MAP, "rt", encoding="utf-8", newline="") as handle:
            direct = {item["h5ad_gene"]: item["orthogroup"] for item in csv.DictReader(handle, delimiter="\t")}
        for position, gene in enumerate(map(str, adata.var_names)):
            orthogroup = direct.get(gene, "")
            if orthogroup in index:
                result[position] = index[orthogroup]
        return result
    species_key = ogbase.SPECIES_KEY[row["species"]]
    dataset_map = safe_maps[row["dataset_id"]]
    species_ogs = gene_to_og[species_key]
    for position, gene in enumerate(map(str, adata.var_names)):
        orthogroup = species_ogs.get(dataset_map.get(gene), "")
        if orthogroup in index:
            result[position] = index[orthogroup]
    return result


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = Path(cfg["feature_cache"])
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite Phase 16 feature cache: {out}")
    out.mkdir(parents=True)
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    allowed = set(cfg["development_species"] + cfg["external_species"])
    rows = [row for row in rows if row["species"] in allowed and row["role"] in {"train", "species_holdout"}]
    if len(rows) != 7 or {row["species"] for row in rows} != allowed:
        raise RuntimeError("Expected exactly five development and two authorized external datasets")
    training_keys = {ogbase.SPECIES_KEY[species] for species in cfg["development_species"]}
    vocabulary, gene_to_og = ogbase.parse_orthogroups(training_keys)
    index = {orthogroup: position for position, orthogroup in enumerate(vocabulary)}
    safe_maps = ogbase.read_safe_gene_mappings()
    entries = []
    for row in rows:
        adata = ad.read_h5ad(row["source_h5ad"], backed="r")
        try:
            mapping = mapped_features(adata, row, safe_maps, gene_to_og, index)
            # No obs columns are accessed: labels remain unopened in this stage.
            obs_names = np.asarray(adata.obs_names.astype(str))
            blocks = []
            block_size = int(cfg["feature_cache_block_cells"])
            for start in range(0, adata.n_obs, block_size):
                stop = min(start + block_size, adata.n_obs)
                block = adata[start:stop, :].to_memory().X
                blocks.append(ogbase.rank_to_orthogroup(block, mapping, len(vocabulary), 256))
        finally:
            adata.file.close()
        features = sparse.vstack(blocks, format="csr")
        slug = row["species"].lower().replace(" ", "_")
        matrix_path = out / f"{slug}.rank_top256_binary.npz"
        obs_path = out / f"{slug}.obs_names.csv.gz"
        sparse.save_npz(matrix_path, features, compressed=True)
        pd.DataFrame({"obs_name": obs_names}).to_csv(obs_path, index=False, compression="gzip")
        entries.append({
            "species": row["species"], "dataset_id": row["dataset_id"],
            "kind": "development" if row["role"] == "train" else "external",
            "cells": int(features.shape[0]), "features": int(features.shape[1]),
            "mapped_input_genes": int((mapping >= 0).sum()),
            "mean_active_features": float(features.getnnz(axis=1).mean()),
            "matrix_file": str(matrix_path), "obs_file": str(obs_path),
            "matrix_sha256": sha256(matrix_path), "obs_sha256": sha256(obs_path),
            "source_h5ad_sha256": sha256(Path(row["source_h5ad"])),
        })
    vocabulary_path = out / "orthogroup_vocabulary.txt"
    vocabulary_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    manifest = {
        "run_id": cfg["run_id"], "status": "completed",
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "representation": cfg["representation"], "labels_read": False,
        "vocabulary_file": str(vocabulary_path), "vocabulary_sha256": sha256(vocabulary_path),
        "entries": entries, "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
