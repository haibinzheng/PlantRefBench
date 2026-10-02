#!/usr/bin/env python3
"""Build immutable development-species orthogroup count caches for scVI baselines."""

from __future__ import annotations

import csv
import argparse
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
import run_phase2_loso_shared_orthogroup as loso  # noqa: E402
import run_phase2_orthogroup_baseline as ogbase  # noqa: E402

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed
    output = Path(f"/data/derived/phylo_plant_fm_pilot/scvi_counts_v1_seed{seed}")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing cache: {output}")
    output.mkdir(parents=True)
    rows_all = list(csv.DictReader(loso.MANIFEST.open(newline="")))
    rows = [row for row in rows_all if row["role"] == "train"]
    mapping = pd.read_csv(loso.LABEL_MAPPING).fillna("")
    label_map = dict(mapping.loc[mapping["mapping_status"].eq("mapped"),
                                 ["source_label", "canonical_family"]].itertuples(index=False, name=None))
    species_keys = {ogbase.SPECIES_KEY[row["species"]] for row in rows}
    vocabulary, gene_to_og = ogbase.parse_orthogroups(species_keys)
    feature_index = {og: index for index, og in enumerate(vocabulary)}
    safe_maps = ogbase.read_safe_gene_mappings()
    (output / "orthogroup_vocabulary.txt").write_text("\n".join(vocabulary) + "\n")

    manifest = []
    for order, row in enumerate(rows):
        adata = ad.read_h5ad(row["source_h5ad"], backed="r")
        try:
            sample_seed = seed + 1009 * order
            positions, labels, obs_hash = loso.sampled_rows(adata, row, label_map, sample_seed)
            dataset_map = safe_maps[row["dataset_id"]]
            species_ogs = gene_to_og[ogbase.SPECIES_KEY[row["species"]]]
            gene_positions, og_positions = [], []
            for position, h5_gene in enumerate(map(str, adata.var_names)):
                reference_gene = dataset_map.get(h5_gene)
                orthogroup = species_ogs.get(reference_gene, "")
                if orthogroup in feature_index:
                    gene_positions.append(position)
                    og_positions.append(feature_index[orthogroup])
            matrix = sparse.csr_matrix(adata[positions, gene_positions].to_memory().X)
            projection = sparse.csr_matrix(
                (
                    np.ones(len(gene_positions), dtype=np.int64),
                    (np.arange(len(gene_positions)), np.asarray(og_positions)),
                ),
                shape=(len(gene_positions), len(vocabulary)),
            )
            counts = (matrix @ projection).tocsr()
            counts.eliminate_zeros()
            nonzero_cells = int(np.count_nonzero(np.diff(counts.indptr)))
            nonzero_cell_fraction = nonzero_cells / max(1, counts.shape[0])
            if not gene_positions or counts.nnz == 0 or nonzero_cell_fraction < 0.99:
                raise RuntimeError(
                    f"Orthogroup coverage gate failed for {row['dataset_id']}: "
                    f"mapped_genes={len(gene_positions)}, nnz={counts.nnz}, "
                    f"nonzero_cells={nonzero_cells}/{counts.shape[0]}"
                )
            if counts.data.size and ((counts.data < 0).any() or not np.equal(counts.data, np.rint(counts.data)).all()):
                raise RuntimeError(f"Non-count values after aggregation for {row['dataset_id']}")
            counts.data = counts.data.astype(np.int64, copy=False)
            obs_names = np.asarray(adata.obs_names.astype(str))[positions]
        finally:
            adata.file.close()

        stem = row["dataset_id"].replace(" ", "_")
        matrix_path = output / f"{stem}.counts.npz"
        obs_path = output / f"{stem}.obs.csv.gz"
        sparse.save_npz(matrix_path, counts, compressed=True)
        pd.DataFrame(
            {"obs_name": obs_names, "canonical_family": labels, "species": row["species"],
             "dataset_id": row["dataset_id"]}
        ).to_csv(obs_path, index=False, compression="gzip")
        details = {
            "dataset_id": row["dataset_id"], "species": row["species"], "sampling_seed": sample_seed,
            "cells": int(counts.shape[0]), "features": int(counts.shape[1]), "nnz": int(counts.nnz),
            "nonzero_cells": nonzero_cells, "nonzero_cell_fraction": nonzero_cell_fraction,
            "mapped_input_genes": len(gene_positions), "sampled_obs_names_sha256": obs_hash,
            "matrix_file": str(matrix_path), "obs_file": str(obs_path),
            "matrix_sha256": sha256(matrix_path), "obs_sha256": sha256(obs_path),
            "library_min": int(np.asarray(counts.sum(axis=1)).min()),
            "library_median": float(np.median(np.asarray(counts.sum(axis=1)))),
            "library_max": int(np.asarray(counts.sum(axis=1)).max()),
        }
        manifest.append(details)
        print(json.dumps(details), flush=True)
    summary = {
        "run_id": f"scvi_counts_v1_seed{seed}", "completed_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "development_species_only": True, "frozen_holdouts_touched": False,
        "orthogroup_features": len(vocabulary), "datasets": manifest,
        "input_sha256": {"protocol": sha256(loso.PROTOCOL), "pilot_manifest": sha256(loso.MANIFEST),
                         "label_mapping": sha256(loso.LABEL_MAPPING), "code": sha256(Path(__file__))},
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed": str(output), "datasets": len(manifest)}), flush=True)


if __name__ == "__main__":
    main()
