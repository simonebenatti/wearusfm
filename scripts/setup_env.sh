#!/bin/bash
module purge
# TODO: completare dopo `module av` su Leonardo
# ml profile/deeplrn
# module load python/3.11.6--gcc--8.5.0 cuda/12.1
# source $WORK/venv_wearusfm/bin/activate

export HF_HOME=$FAST/hf
export TORCH_HOME=$FAST/torch
export XDG_CACHE_HOME=$WORK/.cache
export PIP_CACHE_DIR=$WORK/.cache/pip
