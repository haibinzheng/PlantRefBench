#!/usr/bin/env bash
# Offline-only SAMap overlay installation. It never contacts PyPI or NCBI.
set -euo pipefail

project=/workspace/projects/phylo_plant_fm_pilot
base_python="$project/venvs/scvi_v1/bin/python"
overlay="$project/venvs/samap_overlay_v3"
tools="$project/tools"
blast_name='ncbi-blast-2.17.0+'
wheelhouse=${SAMAP_WHEELHOUSE:-$project/offline_wheels/samap_v3}
blast_archive=${SAMAP_BLAST_ARCHIVE:-$wheelhouse/${blast_name}-x64-linux.tar.gz}
report=/data/reports/phylo_plant_fm_pilot/phase17_samap_smoke_v1/environment

[[ -x "$base_python" ]] || { echo "Missing existing scvi_v1 Python: $base_python" >&2; exit 2; }
[[ -d "$wheelhouse" && -f "$blast_archive" ]] || {
  echo "Provide an uploaded wheelhouse and NCBI BLAST archive; this script is offline-only." >&2; exit 2;
}
[[ ! -e "$overlay" && ! -e "$tools/$blast_name" && ! -e "$report" ]] || {
  echo "Refusing to overwrite an existing SAMap overlay, BLAST installation, or environment report." >&2; exit 2;
}
mkdir -p "$overlay" "$tools" "$report"

# Only the missing SAMap components are placed ahead of scvi_v1. Scientific
# dependencies (including NumPy) remain frozen in scvi_v1. hnswlib is built
# without an isolated environment because the offline build otherwise tries to
# download a duplicate NumPy solely as a build dependency.
"$base_python" -m pip install --no-index --find-links "$wheelhouse" --target "$overlay" --no-deps \
  'sc-samap==3.0.0' dill pybind11 wheel
PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}" "$base_python" -m pip install \
  --no-index --find-links "$wheelhouse" --target "$overlay" --no-deps --no-build-isolation \
  'hnswlib==0.8.0'
tar -xzf "$blast_archive" -C "$tools"
[[ -x "$tools/$blast_name/bin/blastp" && -x "$tools/$blast_name/bin/makeblastdb" ]] || {
  echo "Uploaded BLAST archive did not contain expected executables." >&2; exit 3;
}
PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}" "$base_python" - <<'PY' >"$report/samap_version.txt"
import importlib.metadata
import samap
assert importlib.metadata.version("sc-samap") == "3.0.0"
print("sc-samap=" + importlib.metadata.version("sc-samap"))
print("samap_module=" + str(samap.__file__))
PY
PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}" "$base_python" - <<'PY' >"$report/overlay_imports.txt"
import dill, hnswlib
print("dill=" + str(dill.__file__))
print("hnswlib=" + str(hnswlib.__file__))
PY
"$tools/$blast_name/bin/blastp" -version >"$report/blast_version.txt"
"$base_python" -m pip list --path "$overlay" --format=freeze >"$report/overlay_freeze.txt"
sha256sum "$blast_archive" >"$report/blast_archive_sha256.txt"
sha256sum "$report"/samap_version.txt "$report"/overlay_imports.txt "$report"/blast_version.txt \
  "$report"/overlay_freeze.txt "$report"/blast_archive_sha256.txt >"$report/environment_files_sha256.txt"
printf '%s\n' completed >"$report/COMPLETED"
