#!/bin/bash
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source $WORK/venv_wearusfm/bin/activate

export HF_HOME=$FAST/hf
export TORCH_HOME=$FAST/torch
export XDG_CACHE_HOME=$WORK/.cache
export PIP_CACHE_DIR=$WORK/.cache/pip
