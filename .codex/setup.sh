#!/bin/bash
# Codex runs this script before every task.
# Ensures the conda environment is active for all Python tooling.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate latentscore

