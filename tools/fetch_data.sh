#!/usr/bin/env bash
# Download the three MaleCNS v1.0 files FruitOS needs (~1.1 GB) from Janelia's public bucket.
# Data licence: CC BY 4.0 (see DATA_LICENSE.md).
set -euo pipefail
dest="${1:-raw}"
base="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
mkdir -p "$dest"
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather \
         body-neurotransmitters-male-cns-v1.0.feather \
         connectome-weights-male-cns-v1.0-minconf-0.5.feather; do
  curl -fL --retry 5 -C - -o "$dest/$f" "$base/$f"
done
