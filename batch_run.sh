#!/bin/sh
source activate py310

export HF_ENDPOINT=https://hf-mirror.com

python run_with_earlystopping.py $1 $2