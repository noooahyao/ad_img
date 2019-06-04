#!/bin/bash
INPUT_DIR=$1
OUTPUT_DIR=$2
CUDA_VISIBLE_DEVICES=0 python attack.py \
   --input_dir="${INPUT_DIR}"\
   --output_dir="${OUTPUT_DIR}"\
   --batch_size=22
