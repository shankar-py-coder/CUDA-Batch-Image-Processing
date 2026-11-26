#!/bin/bash

set -e

INPUT_DIR=${1:-input_images}
OUTPUT_DIR=${2:-output_images}

echo "Starting CUDA Batch Image Processing..."

make

./image_processing "$INPUT_DIR" "$OUTPUT_DIR"

echo "Image processing completed!"
