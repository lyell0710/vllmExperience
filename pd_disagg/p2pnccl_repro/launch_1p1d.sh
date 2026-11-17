#!/bin/bash
# R0-4 动态复现：0.17.1 P2pNccl 1P1D（用本机 Qwen2-7B，双卡，不依赖 HF token）
# 精简自官方 disagg_example_p2p_nccl_xpyd.sh。proxy: http 10001 / zmq 30001。
set -u
REPRO=/root/projects/vllm/experiments/pd_disagg/p2pnccl_repro
VENV=/root/venvs/v0.17.1
MODEL=Qwen/Qwen2-7B-Instruct
PROXY_PORT=30001

$VENV/bin/python $REPRO/disagg_proxy_p2p_nccl_xpyd.py > $REPRO/repro_proxy.log 2>&1 &
echo "proxy pid $!"

CUDA_VISIBLE_DEVICES=0 $VENV/bin/vllm serve "$MODEL" \
  --enforce-eager --host 0.0.0.0 --port 20003 --tensor-parallel-size 1 \
  --seed 1024 --dtype float16 --max-model-len 10000 --max-num-batched-tokens 10000 \
  --max-num-seqs 256 --gpu-memory-utilization 0.9 \
  --kv-transfer-config \
  "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"21001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"20003\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
  > $REPRO/repro_prefill.log 2>&1 &
echo "prefill pid $!"

CUDA_VISIBLE_DEVICES=1 $VENV/bin/vllm serve "$MODEL" \
  --enforce-eager --host 0.0.0.0 --port 20005 --tensor-parallel-size 1 \
  --seed 1024 --dtype float16 --max-model-len 10000 --max-num-batched-tokens 10000 \
  --max-num-seqs 256 --gpu-memory-utilization 0.7 \
  --kv-transfer-config \
  "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"22001\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"20005\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
  > $REPRO/repro_decode.log 2>&1 &
echo "decode pid $!"
