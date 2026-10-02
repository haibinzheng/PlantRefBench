#!/usr/bin/env python3
"""Prepare and finalize projection of Catharanthus genes into frozen Phase 12 orthogroups."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import anndata as ad


ROOT = Path("/data/derived/phylo_plant_fm_pilot/external_validation_v1/catharanthus_mapping_v1")
INPUT = ROOT / "ncbi_GCA_024505715.1"
FAA = INPUT / "protein.faa"
GFF = INPUT / "genomic.gff"
LONGEST = ROOT / "catharanthus_roseus.longest_protein.fa"
REFERENCE = ROOT / "five_development_species.longest_protein.fa"
HITS = ROOT / "catharanthus_to_development.diamond.tsv"
ORTHOGROUPS = Path("/data/derived/phylo_plant_fm_pilot/orthology_runs/orthofinder_2_5_5_six_species_v1/Results_Sep10/Orthogroups/Orthogroups.tsv")
H5AD = Path("/data/datasets/Bioinformatics/h5ad/Catharanthus roseus_PRJNA847226_Root.h5ad")
ADDENDUM = Path("/workspace/projects/phylo_plant_fm_pilot/configs/external_validation_v1/catharanthus_mapping_addendum_v1.json")
DEV_FASTAS = [
    Path("/data/derived/phylo_plant_fm_pilot/orthology_inputs/reference_longest_protein_v1/arabidopsis_thaliana.longest_protein.fa"),
    Path("/data/derived/phylo_plant_fm_pilot/orthology_inputs/reference_longest_protein_v1/oryza_sativa.longest_protein.fa"),
    Path("/data/derived/phylo_plant_fm_pilot/orthology_inputs/reference_longest_protein_v1/zea_mays.longest_protein.fa"),
    Path("/data/derived/phylo_plant_fm_pilot/orthology_inputs/reference_longest_protein_v1/glycine_max.longest_protein.fa"),
    Path("/data/derived/phylo_plant_fm_pilot/orthology_inputs/reference_longest_protein_v1/medicago_truncatula.longest_protein.fa"),
]
DEV_KEYS = {"arabidopsis_thaliana", "oryza_sativa", "zea_mays", "glycine_max", "medicago_truncatula"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fasta_records(path):
    header, parts = None, []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not ADDENDUM.exists():
        raise FileNotFoundError("Frozen mapping addendum is required")
    chosen = {}
    for header, sequence in fasta_records(FAA):
        protein = header.split()[0]
        match = re.search(r"M9H77_\d+", header)
        if not match:
            continue
        gene = match.group(0)
        candidate = (len(sequence), protein, sequence)
        current = chosen.get(gene)
        if current is None or candidate[0] > current[0] or (candidate[0] == current[0] and protein < current[1]):
            chosen[gene] = candidate
    with LONGEST.open("w", encoding="utf-8") as handle:
        for gene in sorted(chosen):
            length, protein, sequence = chosen[gene]
            handle.write(f">catharanthus_roseus|{gene} protein={protein} length={length}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")
    with REFERENCE.open("w", encoding="utf-8") as output:
        for path in DEV_FASTAS:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    output.write(line)
    manifest = {
        "assembly": "GCA_024505715.1", "protein_records": sum(1 for _ in fasta_records(FAA)),
        "selected_longest_gene_proteins": len(chosen), "protein_faa_sha256": sha256(FAA),
        "genomic_gff_sha256": sha256(GFF), "longest_fasta_sha256": sha256(LONGEST),
        "development_reference_sha256": sha256(REFERENCE), "mapping_addendum_sha256": sha256(ADDENDUM),
    }
    (ROOT / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def eligible_reference_gene_to_og():
    result = {}
    eligible = set()
    with ORTHOGROUPS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = {column: column.removesuffix(".longest_protein") for column in reader.fieldnames[1:]}
        for row in reader:
            if sum(bool(row[column].strip()) for column, species in columns.items() if species in DEV_KEYS) < 2:
                continue
            og = row["Orthogroup"]
            eligible.add(og)
            for column, species in columns.items():
                if species not in DEV_KEYS:
                    continue
                for item in row[column].split(","):
                    item = item.strip()
                    if item:
                        result[item] = og
                        result[item.split("|", 1)[-1]] = og
    return result, eligible


def finalize():
    reference_to_og, eligible = eligible_reference_gene_to_og()
    best = {}
    retained_hits = 0
    with HITS.open(encoding="utf-8", newline="") as handle:
        fields = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qcovhsp", "scovhsp"]
        for row in csv.DictReader(handle, delimiter="\t", fieldnames=fields):
            if float(row["evalue"]) > 1e-5 or float(row["qcovhsp"]) < 50 or float(row["scovhsp"]) < 50:
                continue
            og = reference_to_og.get(row["sseqid"])
            if og not in eligible:
                continue
            retained_hits += 1
            gene = row["qseqid"].split("|", 1)[-1]
            key = (-float(row["bitscore"]), float(row["evalue"]), row["sseqid"])
            if gene not in best or key < best[gene][0]:
                best[gene] = (key, og, row["sseqid"], row)
    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        h5ad_genes = list(map(str, adata.var_names))
    finally:
        adata.file.close()
    rows = []
    for h5ad_gene in h5ad_genes:
        locus = h5ad_gene.replace("-", "_")
        if locus in best:
            _, og, subject, hit = best[locus]
            rows.append({"h5ad_gene": h5ad_gene, "locus_tag": locus, "orthogroup": og,
                         "subject_gene": subject, "bitscore": hit["bitscore"], "evalue": hit["evalue"],
                         "qcovhsp": hit["qcovhsp"], "scovhsp": hit["scovhsp"]})
    mapping_path = ROOT / "catharanthus_h5ad_to_frozen_orthogroup.tsv.gz"
    with gzip.open(mapping_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]) if rows else ["h5ad_gene"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "query_genes_with_retained_hit": len(best), "retained_hits": retained_hits,
        "h5ad_genes": len(h5ad_genes), "h5ad_genes_mapped": len(rows),
        "h5ad_mapping_coverage": len(rows) / len(h5ad_genes),
        "eligible_frozen_orthogroups": len(eligible), "mapping_sha256": sha256(mapping_path),
        "mapping_gate_50pct": len(rows) / len(h5ad_genes) >= 0.50,
        "target_expression_or_labels_used": False,
    }
    (ROOT / "mapping_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "finalize"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else finalize()
