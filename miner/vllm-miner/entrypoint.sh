#!/bin/bash
set -e

# Auto-detect GPUs and set CUDA_VISIBLE_DEVICES if not already set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
        export CUDA_VISIBLE_DEVICES
        echo "Auto-detected $GPU_COUNT GPUs, setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
fi

echo "Starting pearl-gateway..."
pearl-gateway start &
PEARL_PID=$!

# Wait until the gateway is ready
curl -s http://localhost:8339/metrics --retry-delay 1 --retry 20 --retry-all-errors > /dev/null

echo "Starting vllm serve with args: $@"
exec vllm serve "$@"
