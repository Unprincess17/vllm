#!/bin/bash

# benchmark the overhead of disaggregated prefill.
# methodology:
# - send all request to prefill vLLM instance. It will buffer KV cache.
# - then send all request to decode instance. 
# - The TTFT of decode instance is the overhead.

set -ex

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  sleep 10

  # remove vllm config file
  rm -rf ~/.config/vllm

  # Print the GPU memory usage
  # so that we know if all GPU processes are killed.
  gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  # The memory usage should be 0 MB.
  echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


benchmark() {

  # export VLLM_LOGGING_LEVEL=DEBUG
  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  # compare chunked prefill with disaggregated prefill

  results_folder="./results"
  model="LLM-Research/Llama-3.2-1B-Instruct"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=10
  qps=$1
  prefix_len=50
  input_len=2048
  output_len=$2
  benchmark_type=$3



  UCX_TLS="cuda_ipc,cuda_copy,tcp" \
  LMCACHE_CONFIG_FILE="/root/vllm/examples/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml" \
  LMCACHE_LOG_LEVEL="DEBUG" \
  LMCACHE_USE_EXPERIMENTAL="True" \
  LMCACHE_USE_LAYERAWARE="True" \
  VLLM_SERVER_DEV_MODE="1" \
  VLLM_ENABLE_V1_MULTIPROCESSING="1" \
  VLLM_USE_V1="1" \
  VLLM_WORKER_MULTIPROC_METHOD="spawn" \
  CUDA_VISIBLE_DEVICES="2" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --disable-log-requests \
    --enforce-eager \
    --skip-logits \
    --kv-transfer-config \
    "{\"kv_connector\":\"LMCacheConnectorV2\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\": {\"discard_partial_chunks\": false, \"lmcache_rpc_port\": \"producer1\"}}" &
    

  UCX_TLS="cuda_ipc,cuda_copy,tcp" \
  LMCACHE_CONFIG_FILE="/root/vllm/examples/lmcache/disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml" \
  LMCACHE_LOG_LEVEL="DEBUG" \
  LMCACHE_USE_EXPERIMENTAL="True" \
  LMCACHE_USE_LAYERAWARE="True" \
  VLLM_SERVER_DEV_MODE="1" \
  VLLM_ENABLE_V1_MULTIPROCESSING="1" \
  VLLM_USE_V1="1" \
  VLLM_WORKER_MULTIPROC_METHOD="spawn" \
  CUDA_VISIBLE_DEVICES="3" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --disable-log-requests \
    --enforce-eager \
    --gpu-memory-utilization 0.6 \
    --skip-logits \
    --kv-transfer-config \
    "{\"kv_connector\":\"LMCacheConnectorV2\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\": {\"discard_partial_chunks\": false, \"lmcache_rpc_port\": \"consumer1\"}}" &

  wait_for_server 8100
  wait_for_server 8200

  python3 ./dual_benchmark_serving.py $benchmark_type

  kill_gpu_processes
}


main() {
  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd layeraware_benchmarks

  rm -rf results
  mkdir results

  default_qps=10
  default_output_len=1
  # benchmark type: simu, seq
  benchmark_type=${1:-"simu"}
  benchmark $default_qps $default_output_len $benchmark_type

}


main "$@"
