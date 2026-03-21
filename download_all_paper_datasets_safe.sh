#!/bin/bash
# Safe re-download script for all SGFormer paper datasets (11 total)
# Sources used are the same ones already used in this workspace:
# - Planetoid: PyG auto download (inline in this script)
# - Medium non-Planetoid: Google Drive + yandex-research filtered splits
# - Large OGB: OGB auto download via one-epoch runs

set -euo pipefail

ROOT_DIR="/Users/vn59a0h/thesis/PCGT"
DATA_DIR="$ROOT_DIR/data"
MEDIUM_DIR="$ROOT_DIR/medium"
LARGE_DIR="$ROOT_DIR/large"

# PyTorch 2.6+ compatibility for legacy OGB processed files.
# OGB loaders rely on torch.load() for non-weights data objects.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

warn() {
  echo "[WARN] $*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

cleanup_placeholder_dirs() {
  # Legacy placeholder dirs can appear from older workflows.
  # Keep active dataset paths only; remove placeholders if they are empty.
  local placeholders=(
    "$DATA_DIR/actor"
    "$DATA_DIR/chameleon"
    "$DATA_DIR/film"
    "$DATA_DIR/squirrel"
    "$DATA_DIR/pokec/split_0.5_0.25"
  )

  for d in "${placeholders[@]}"; do
    if [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
      rmdir "$d" && log "Removed placeholder dir: ${d#$ROOT_DIR/}"
    fi
  done
}

download_planetoid_if_missing() {
  local cora="$DATA_DIR/Planetoid/cora/raw/ind.cora.x"
  local citeseer="$DATA_DIR/Planetoid/citeseer/raw/ind.citeseer.x"
  local pubmed="$DATA_DIR/Planetoid/pubmed/raw/ind.pubmed.x"

  if [ -f "$cora" ] && [ -f "$citeseer" ] && [ -f "$pubmed" ]; then
    log "Planetoid already cached; skipping"
    return 0
  fi

  log "Downloading Planetoid datasets with PyG (cora/citeseer/pubmed)"
  python - <<PY
from torch_geometric.datasets import Planetoid
root = r"$DATA_DIR/Planetoid"
for name in ["cora", "citeseer", "pubmed"]:
    ds = Planetoid(root=root, name=name)
    d = ds[0]
    print(f"ready {name}: nodes={d.num_nodes} edges={d.num_edges} feats={d.num_node_features} classes={ds.num_classes}")
PY
}

download_by_id_if_missing() {
  local file_id="$1"
  local target="$2"
  local label="$3"
  if [ -f "$target" ]; then
    log "Already present: $label"
    return 0
  fi
  mkdir -p "$(dirname "$target")"
  log "Downloading: $label"
  if ! gdown "https://drive.google.com/uc?id=${file_id}" -O "$target" >/dev/null 2>&1; then
    warn "Failed: $label"
    return 1
  fi
  log "Done: $label"
}

run_or_warn() {
  local desc="$1"
  shift
  log "$desc"
  if ! "$@"; then
    warn "Command failed: $desc"
    return 1
  fi
  return 0
}

main() {
  log "Starting safe dataset re-download for SGFormer paper"
  mkdir -p "$DATA_DIR"

  if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/venv/bin/activate"
    log "Activated venv"
  else
    warn "venv not found at $ROOT_DIR/venv (continuing with current python)"
  fi

  if ! have_cmd python; then
    echo "python command not found" >&2
    exit 1
  fi

  if ! have_cmd gdown; then
    log "Installing gdown"
    python -m pip install -q gdown
  fi

  if ! python -c "import googledrivedownloader" >/dev/null 2>&1; then
    log "Installing googledrivedownloader"
    python -m pip install -q googledrivedownloader
  fi

  # Provide compatibility module expected by project imports, without changing source files.
  python - <<'PY'
import os, site
site_pkgs = site.getsitepackages()[0]
shim = os.path.join(site_pkgs, 'google_drive_downloader.py')
if not os.path.exists(shim):
    with open(shim, 'w', encoding='utf-8') as f:
        f.write(
            'class GoogleDriveDownloader:\n'
            '    @staticmethod\n'
            '    def download_file_from_google_drive(file_id, dest_path, showsize=True):\n'
            '        import googledrivedownloader as gdd\n'
            '        return gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, showsize=showsize)\n'
        )
PY

  log "Step 1/5: Planetoid datasets (cora/citeseer/pubmed)"
  run_or_warn "Download/check Planetoid inline" download_planetoid_if_missing

  # Prevent duplicate/confusing empty folders from lingering.
  cleanup_placeholder_dirs

  log "Step 2/5: Medium non-Planetoid datasets"
  mkdir -p "$DATA_DIR/deezer" "$DATA_DIR/geom-gcn/film" "$DATA_DIR/wiki_new/chameleon" "$DATA_DIR/wiki_new/squirrel"

  # Google Drive file IDs already validated in this workspace.
  download_by_id_if_missing "1P6w53eYamAPVuI_PVbbJVBidrxJfMl9f" "$DATA_DIR/deezer/deezer-europe.mat" "deezer-europe.mat" || true
  download_by_id_if_missing "1szPPOymVXJibvI3SLCZkYMOFjHAAnhKK" "$DATA_DIR/geom-gcn/film/out1_graph_edges.txt" "film edges" || true
  download_by_id_if_missing "1j8_2DsviL6W2cO4LCsNVo1r0c3htpbOS" "$DATA_DIR/geom-gcn/film/out1_node_feature_label.txt" "film node labels" || true

  # Film split files 0..9
  FILM_SPLIT_IDS=(
    "1EehMEvc_HP4YKmWkra0jLGxAOKHZpu_i"
    "1XdHlGwgM8rjrnG-_dbhGjvvgE3J545qh"
    "1BjDcmtDTlRR0axSY-D_4i7L4_WfAOeIS"
    "1VPVDeLdQ8VJ9kAY-UC6v_RIL9Yd60-Li"
    "1ggi1VwfAy2IbAEpl6zezJF6uMxmfLjrS"
    "1f2Xo2bQFBlc-i5Hwg4tKdcixDcUfpPN_"
    "1AQj5U7R-StOWB7N6hXigcwZ8LyrCUBI4"
    "1qp4oALR17PkwzHPCjswKpM2q8ISHFuP8"
    "1Nm3i5zX0oN_Au9624EZU8iKYlw9zxq8o"
    "1BhWJEdr_b6vmdZFLLlKHDpCCHamURG_D"
  )

  for i in "${!FILM_SPLIT_IDS[@]}"; do
    download_by_id_if_missing "${FILM_SPLIT_IDS[$i]}" "$DATA_DIR/geom-gcn/film/film_split_0.6_0.2_${i}.npz" "film split ${i}" || true
  done

  # Filtered npz files used by medium code for chameleon/squirrel.
  for name in chameleon squirrel; do
    target="$DATA_DIR/wiki_new/$name/${name}_filtered.npz"
    if [ ! -f "$target" ]; then
      log "Downloading ${name}_filtered.npz"
      # Using python requests to avoid depending on curl availability/policy.
      python - <<PY
import requests
url = 'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/${name}_filtered.npz'
out = '${target}'
r = requests.get(url, timeout=120)
r.raise_for_status()
with open(out, 'wb') as f:
    f.write(r.content)
print('saved', out)
PY
    else
      log "Already present: ${name}_filtered.npz"
    fi
  done

  log "Step 3/5: Large datasets download only (no model training)"

  # OGB datasets: use absolute cache path and skip when already present.
  # Also force non-interactive confirmation to avoid EOFError in scripted runs.
  export OGB_ROOT="$DATA_DIR/ogb"
  python - <<'PY'
import os
import builtins
from ogb.nodeproppred import NodePropPredDataset

root = os.environ['OGB_ROOT']
mapping = {
    'ogbn-arxiv': 'ogbn_arxiv',
    'ogbn-proteins': 'ogbn_proteins',
    'ogbn-products': 'ogbn_products',
}

for name, folder in mapping.items():
    processed = os.path.join(root, folder, 'processed', 'data_processed')
    if os.path.exists(processed):
        print(f'checking {name}: already cached at {processed}')
        continue

    print(f'checking {name}: cache missing, downloading...')
    builtins.input = lambda prompt='': 'y'
    NodePropPredDataset(name=name, root=root)

print('ogb datasets ready')
PY

  # Pokec: try folder download only (no training). This can still be restricted by Drive permissions.
  mkdir -p "$DATA_DIR/pokec"
  if [ ! -f "$DATA_DIR/pokec/pokec.mat" ]; then
    log "Downloading pokec folder from Google Drive"
    gdown --folder "https://drive.google.com/drive/folders/1eEbm9qgC-WLJwwTIXBUYFOEdSl5rItTT" --remaining-ok --output "$DATA_DIR/pokec" >/dev/null 2>&1 || true
  fi

  log "Step 4/5: Presence summary"

  ok=0
  pending=0

  check_file() {
    local f="$1"
    local name="$2"
    if [ -f "$f" ]; then
      echo "  [OK] $name"
      ok=$((ok+1))
    else
      echo "  [PENDING] $name"
      pending=$((pending+1))
    fi
  }

  check_file "$DATA_DIR/Planetoid/cora/raw/ind.cora.x" "cora"
  check_file "$DATA_DIR/Planetoid/citeseer/raw/ind.citeseer.x" "citeseer"
  check_file "$DATA_DIR/Planetoid/pubmed/raw/ind.pubmed.x" "pubmed"
  check_file "$DATA_DIR/geom-gcn/film/out1_graph_edges.txt" "film"
  check_file "$DATA_DIR/deezer/deezer-europe.mat" "deezer-europe"
  check_file "$DATA_DIR/wiki_new/squirrel/squirrel_filtered.npz" "squirrel"
  check_file "$DATA_DIR/wiki_new/chameleon/chameleon_filtered.npz" "chameleon"
  check_file "$DATA_DIR/ogb/ogbn_arxiv/processed/data_processed" "ogbn-arxiv"
  check_file "$DATA_DIR/ogb/ogbn_proteins/processed/data_processed" "ogbn-proteins"
  check_file "$DATA_DIR/ogb/ogbn_products/processed/data_processed" "amazon2m"

  # pokec validity check: mat must exist and not be tiny placeholder
  if [ -f "$DATA_DIR/pokec/pokec.mat" ]; then
    sz=$(python - <<'PY'
import os
print(os.path.getsize('data/pokec/pokec.mat'))
PY
)
    if [ "$sz" -gt 1000000 ]; then
      echo "  [OK] pokec"
      ok=$((ok+1))
    else
      echo "  [PENDING] pokec (found tiny file: ${sz} bytes)"
      pending=$((pending+1))
    fi
  else
    echo "  [PENDING] pokec"
    pending=$((pending+1))
  fi

  log "Step 5/5: Completed"
  echo "Summary: OK=${ok}, PENDING=${pending}"
  if [ "$pending" -gt 0 ]; then
    echo "Some datasets are pending (most commonly pokec due Drive access limits)."
    echo "If pokec is pending, manually download valid pokec files into $DATA_DIR/pokec and rerun this script."
  fi

  # Final cleanup in case any helper step recreated placeholders.
  cleanup_placeholder_dirs
}

main "$@"
