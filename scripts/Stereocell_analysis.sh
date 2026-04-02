#!/bin/bash

# conda activate /opt/miniforge3/envs/env-cellbinv2

CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
    -c SN \
    -p cellbin2/config/demos/Stereocell_analysis.json \
    -o test/SN