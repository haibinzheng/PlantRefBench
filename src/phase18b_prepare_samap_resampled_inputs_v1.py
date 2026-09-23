#!/usr/bin/env python3
"""Prepare one seed of reference-resampled, label-free external SAMap inputs."""

from __future__ import annotations

import csv
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT = Path("/workspace/projects/phylo_plant_fm_pilot")
sys.path.insert(0, str(PROJECT / "src"))
import run_phase2_orthogroup_baseline as ogbase  # noqa: E402

MANIFEST = PROJECT / "configs/pilot_manifest.csv"
CATH_MAP = Path("/data/derived/phylo_plant_fm_pilot/external_validation_v1/catharanthus_mapping_v1/catharanthus_h5ad_to_frozen_orthogroup.tsv.gz")
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


def fasta_ids(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                identifier = line[1:].split(maxsplit=1)[0]
                gene = identifier.split("|", 1)[-1]
                if gene in result and result[gene] != identifier:
                    raise RuntimeError(f"Ambiguous FASTA gene {gene}")
                result[gene] = identifier
    return result


def cath_gene_map() -> dict[str, str]:
    with gzip.open(CATH_MAP, "rt", encoding="utf-8-sig", newline="") as f:
        return {row["h5ad_gene"]: row["locus_tag"] for row in csv.DictReader(f, delimiter="\t")}


def canonical_lookup() -> dict[str, str]:
    mapping = pd.read_csv(LABEL_MAPPING)
    mapped = mapping.loc[mapping["mapping_status"].eq("mapped"), ["source_label", "canonical_family"]]
    return dict(mapped.itertuples(index=False, name=None))


def selected_positions(n: int, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(n)[: min(cap, n)])


def prepare_one(code: str, cap: int, seed: int, cfg: dict, safe_maps: dict, cath_map: dict,
                out: Path, hash_cache: dict[str, str]) -> dict:
    spec = cfg["species"][code]
    row = manifest_row(spec["species"], spec["role"])
    fasta = Path(spec["protein_fasta"])
    protein_ids = fasta_ids(fasta)
    adata = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        selected = selected_positions(adata.n_obs, cap, seed)
        obs_names = np.asarray(adata.obs_names.astype(str))[selected]
        matrix = sparse.csr_matrix(adata[selected, :].to_memory().X)
        genes = np.asarray(adata.var_names.astype(str))
        total_cells = int(adata.n_obs)
    finally:
        adata.file.close()
    gene_map = ({gene: cath_map.get(gene, gene.replace("-", "_")) for gene in genes}
                if code == "cr" else safe_maps[row["dataset_id"]])
    mapped = [(i, protein_ids.get(gene_map.get(gene, ""), "")) for i, gene in enumerate(genes)]
    positions = np.asarray([i for i, protein in mapped if protein], dtype=int)
    proteins = [protein for _, protein in mapped if protein]
    coverage = len(positions) / max(len(genes), 1)
    if coverage < 0.70:
        raise RuntimeError(f"{code} gene-map coverage below gate: {coverage:.4f}")
    ordered = sorted(set(proteins))
    index = {protein: i for i, protein in enumerate(ordered)}
    projection = sparse.csr_matrix(
        (np.ones(len(positions), dtype=np.int8),
         (np.arange(len(positions)), [index[p] for p in proteins])),
        shape=(len(positions), len(ordered)),
    )
    compressed = (matrix[:, positions] @ projection).tocsr()
    active = np.asarray(matrix.getnnz(axis=1)).ravel()
    mapped_active = np.asarray(matrix[:, positions].getnnz(axis=1)).ravel()
    active_coverage = float(np.mean(mapped_active / np.maximum(active, 1)))
    if active_coverage < 0.70:
        raise RuntimeError(f"{code} active-feature coverage below gate: {active_coverage:.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=compressed, obs=pd.DataFrame(index=obs_names),
               var=pd.DataFrame(index=np.asarray(ordered))).write_h5ad(out, compression="gzip")
    source = str(row["source_h5ad"])
    if source not in hash_cache:
        hash_cache[source] = sha256(Path(source))
    return {
        "code": code, "species": spec["species"], "dataset_id": row["dataset_id"],
        "selected_cells": int(len(selected)), "input_cells": total_cells,
        "input_genes": int(len(genes)), "mapped_input_genes": int(len(positions)),
        "gene_map_coverage": coverage, "mean_active_feature_coverage": active_coverage,
        "prepared_features": int(len(ordered)), "sampling_seed": int(seed),
        "prepared_h5ad": str(out), "prepared_h5ad_sha256": sha256(out),
        "source_h5ad_sha256": hash_cache[source], "protein_fasta_sha256": sha256(fasta),
    }


def selected_label_vocabulary(code: str, prepared: dict, cfg: dict, lookup: dict[str, str]) -> set[str]:
    spec = cfg["species"][code]
    row = manifest_row(spec["species"], spec["role"])
    selected = ad.read_h5ad(prepared["prepared_h5ad"], backed="r")
    raw = ad.read_h5ad(row["source_h5ad"], backed="r")
    try:
        column = ogbase.source_label_column(raw)
        labels = pd.Series(raw.obs[column].astype(str).map(lookup).to_numpy(),
                           index=raw.obs_names.astype(str))
        values = labels.reindex(selected.obs_names.astype(str)).dropna()
        return set(values.astype(str))
    finally:
        selected.file.close()
        raw.file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    stability = json.loads(config_path.read_text(encoding="utf-8"))
    if args.seed not in stability["new_seeds"]:
        raise RuntimeError("Seed is not an authorized new SAMap resampling seed")
    base_config_path = Path(stability["base_config"])
    cfg = json.loads(base_config_path.read_text(encoding="utf-8"))
    addendum = Path(stability["input_addendum"])
    original_manifest = json.loads(Path(stability["reused_input_manifest"]).read_text(encoding="utf-8"))
    maps_manifest = json.loads(Path(stability["maps_manifest"]).read_text(encoding="utf-8"))
    if maps_manifest.get("pair_count") != 20:
        raise RuntimeError("Frozen reciprocal-map manifest is incomplete")
    root = Path(stability["prepared_root"]) / f"seed{args.seed}" / "inputs"
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite prepared inputs: {root}")
    safe_maps = ogbase.read_safe_gene_mappings()
    c_map = cath_gene_map()
    hashes: dict[str, str] = {}
    lookup = canonical_lookup()
    overall = {
        "run_id": stability["run_id"], "seed": int(args.seed),
        "labels_in_samap_inputs": False, "target_cells_reused_from_phase17": True,
        "maps_reused": True, "targets": {},
    }
    for target_code, target_cfg in cfg["targets"].items():
        target_result = {"conditions": {}}
        vocabularies: dict[str, set[str]] = {}
        original_target = original_manifest["targets"][target_code]["conditions"]["all_sources"]["inputs"][target_code]
        target_shared = dict(original_target)
        target_path = Path(target_shared["prepared_h5ad"])
        if sha256(target_path) != target_shared["prepared_h5ad_sha256"]:
            raise RuntimeError(f"Frozen target input hash mismatch: {target_code}")
        for condition, condition_cfg in target_cfg["conditions"].items():
            condition_root = root / target_code / condition
            items = {}
            for source_code in condition_cfg["sources"]:
                seed = (
                    int(cfg["source_sampling_seeds"][source_code])
                    + 100000 * (int(args.seed) - int(stability["reused_seed"]))
                    + (0 if target_code == "sb" else 1000)
                )
                items[source_code] = prepare_one(
                    source_code, int(condition_cfg["cells_per_source"]), seed, cfg,
                    safe_maps, c_map, condition_root / f"{source_code}.h5ad", hashes,
                )
            items[target_code] = target_shared
            source_total = sum(items[code]["selected_cells"] for code in condition_cfg["sources"])
            if source_total != int(target_cfg["source_total_budget"]):
                raise RuntimeError(f"Source budget mismatch for {target_code}/{condition}: {source_total}")
            vocab = set().union(*(selected_label_vocabulary(code, items[code], cfg, lookup)
                                  for code in condition_cfg["sources"]))
            vocabularies[condition] = vocab
            target_result["conditions"][condition] = {
                "sources": condition_cfg["sources"], "source_total_cells": source_total,
                "inputs": items,
            }
        common = sorted(set.intersection(*vocabularies.values()))
        if not common:
            raise RuntimeError(f"No common source vocabulary for {target_code}")
        target_result["common_source_vocabulary"] = common
        target_result["target_obs_identical_across_conditions"] = True
        if not target_result["target_obs_identical_across_conditions"]:
            raise RuntimeError(f"Target inputs differ across conditions for {target_code}")
        overall["targets"][target_code] = target_result
    overall["config_sha256"] = sha256(config_path)
    overall["base_config_sha256"] = sha256(base_config_path)
    overall["input_addendum_sha256"] = sha256(addendum)
    overall["maps_manifest_sha256"] = sha256(Path(stability["maps_manifest"]))
    overall["code_sha256"] = sha256(Path(__file__))
    (root / "manifest.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
