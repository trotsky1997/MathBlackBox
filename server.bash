#!/bin/sh
source activate vllmserver

export HF_ENDPOINT=https://hf-mirror.com

echo $HOSTNAME,$2,$1 >> server.csv
if [ -z "$3" ]
then
  python -m vllm.entrypoints.openai.api_server --model $1 --port $2 --trust-remote-code
else
  python -m vllm.entrypoints.openai.api_server --model $1 --port $2 --enable-lora --lora-modules test-lora=$3 --trust-remote-code
fi