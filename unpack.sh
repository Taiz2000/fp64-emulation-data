#!/usr/bin/env bash
# Restore the per-run cuBLASLt logs (lt/) and per-call kernel dumps (kern/) from their archives.
# The analysis scripts read some of these files, so run this once after cloning.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
find "$ROOT/results/final" -name '*.tar.xz' \( -name lt.tar.xz -o -name kern.tar.xz \) | while read -r a; do
  tar -C "$(dirname "$a")" -xJf "$a"
done
echo "unpacked lt/ and kern/ under results/final (about 2.3 GB)"
