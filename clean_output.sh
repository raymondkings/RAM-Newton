#!/usr/bin/env bash

# Deletes all .csv, .png, and .mp4 files from the output/ directory next to this script.

set -euo pipefail

OUTPUT_DIR="$(cd "$(dirname "$0")" && pwd)/output"

find "$OUTPUT_DIR" \
    \( -name "*.csv" -o -name "*.png" -o -name "*.mp4" \) \
    -delete

echo "Output folder cleaned."
