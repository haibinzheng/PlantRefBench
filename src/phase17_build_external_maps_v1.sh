#!/usr/bin/env bash
# Build the 20 unique reciprocal protein-map pairs needed by the two external matrices.
set -euo pipefail

project=/workspace/projects/phylo_plant_fm_pilot
config="$project/configs/phase17_samap_external_v1.json"
root=/data/derived/phylo_plant_fm_pilot/phase17_samap_external_v1
maps="$root/maps"
log_dir=/data/logs/phylo_plant_fm_pilot/phase17_samap_external_v1/maps
checkpoint=/data/checkpoints/phylo_plant_fm_pilot/phase17_samap_external_v1
python_bin="$project/venvs/scvi_v1/bin/python"
map_script="$project/tools/SAMap-v3.0.0-map_genes.sh"

[[ -f "$config" && -f "$map_script" ]] || exit 2
[[ ! -e "$checkpoint/maps_completed" ]] || { echo "Maps already completed; refusing duplicate run." >&2; exit 2; }
mkdir -p "$maps" "$log_dir" "$checkpoint"
export PATH="$project/tools/ncbi-blast-2.17.0+/bin:$PATH"
printf '%s\n' "$$" > "$log_dir/launcher.pid"

# Reuse the completed, hashed Oryza-Sorghum reciprocal tables.
if [[ ! -e "$maps/ossb" ]]; then
  ln -s /data/derived/phylo_plant_fm_pilot/phase17_samap_smoke_v1/maps/ossb "$maps/ossb"
fi

cd "$root"
"$python_bin" - "$config" "$map_script" <<'PY'
import json, pathlib, subprocess, sys

cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
script = pathlib.Path(sys.argv[2])
maps = pathlib.Path(cfg["prepared_root"]) / "maps"
logs = pathlib.Path("/data/logs/phylo_plant_fm_pilot/phase17_samap_external_v1/maps")
for left, right in cfg["map_pairs"]:
    pair = left + right
    out = maps / pair
    tables = (out / f"{left}_to_{right}.txt", out / f"{right}_to_{left}.txt")
    if all(path.is_file() and path.stat().st_size > 0 for path in tables):
        print(f"REUSE {pair}", flush=True)
        continue
    if out.exists():
        raise SystemExit(f"Partial map directory requires review: {out}")
    a, b = cfg["species"][left], cfg["species"][right]
    command = ["bash", str(script), "--tr1", a["protein_fasta"], "--t1", "prot", "--n1", left,
               "--tr2", b["protein_fasta"], "--t2", "prot", "--n2", right,
               "--threads", str(cfg["map_threads"])]
    with (logs / f"{pair}.log").open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    if not all(path.is_file() and path.stat().st_size > 0 for path in tables):
        raise SystemExit(f"Missing reciprocal tables after {pair}")
    print(f"COMPLETED {pair}", flush=True)
PY

"$python_bin" - "$config" "$maps" <<'PY'
import hashlib, json, pathlib, sys

cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
items = {}
for left, right in cfg["map_pairs"]:
    directory = root / (left + right)
    if not directory.exists():
        directory = root / (right + left)
    for name in (f"{left}_to_{right}.txt", f"{right}_to_{left}.txt"):
        path = directory / name
        items[f"{left}:{right}:{name}"] = {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha(path)}
manifest = {"run_id": cfg["run_id"], "pair_count": len(cfg["map_pairs"]), "tables": items}
(root.parent / "maps_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
touch "$checkpoint/maps_completed"
