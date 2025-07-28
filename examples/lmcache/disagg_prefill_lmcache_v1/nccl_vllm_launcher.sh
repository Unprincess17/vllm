#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <prefiller | decoder> [model]"
    exit 1
fi

if [[ $# -eq 1 ]]; then
    echo "Using default model: meta-llama/Llama-3.2-3B-Instruct"
    MODEL="LLM-Research/Llama-3.2-3B-Instruct"
else
    echo "Using model: $2"
    MODEL=$2
fi


if [[ $1 == "prefiller" ]]; then
    # Prefiller listens on port 8100
    prefill_config_file=$SCRIPT_DIR/configs/lmcache-prefiller-config.yaml

        CUDA_VISIBLE_DEVICES=0 \
	VLLM_USE_V1=0 \
        vllm serve $MODEL \
        --port 8100 \
        --disable-log-requests \
        --enforce-eager \
	--gpu-memory-utilization 0.6 \
        --kv-transfer-config \
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer", "kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}'


elif [[ $1 == "decoder" ]]; then
    # Decoder listens on port 8200
    decode_config_file=$SCRIPT_DIR/configs/lmcache-decoder-config.yaml

        CUDA_VISIBLE_DEVICES=1 \
	VLLM_USE_V1=0 \
        vllm serve $MODEL \
        --port 8200 \
        --disable-log-requests \
        --enforce-eager \
	--gpu-memory-utilization 0.6 \
        --kv-transfer-config \
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}'


else
    echo "Invalid role: $1"
    echo "Should be either prefill, decode"
    exit 1
fi
