#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

PYTHONPATH=$SCRIPT_DIR torchrun --nproc_per_node 1 $(which uvicorn) api.main:app --reload
