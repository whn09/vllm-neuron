#!/bin/bash
# MiMo-V2.5 single-instance server — TP64 with expert parallelism, BF16.
#
# 48 layers / 256 routed experts / top-8 / hidden 4096. At TP=64 with EP=64 each
# rank owns 4 experts, which is what the MoE kernels are sized against below.
#
# Text decoder only. The released checkpoint is multimodal, and its custom
# configuration_mimo_v2.py sets self.vision_config unconditionally, so
# --limit-mm-per-prompt pins every modality to 0 to keep the vision tower out of
# bucket sizing (see MiMoV2ForCausalLM.from_configs and
# NeuronPlatform._mm_disabled for why that is load-bearing rather than cosmetic).
#
# The NEFF cache key bakes in (max_num_seqs, max_model_len), so changing BS or
# SEQ forces a fresh compile (~5-10 min). Keep them as env knobs rather than
# editing inline, and let bench.sh drive load against an already-warm server.
#
# Usage:
#   bash serve_mimo.sh                 # BS=32, SEQ=1024
#   BS=1 SEQ=512 bash serve_mimo.sh

set -x

MODEL="${MODEL:-/opt/dlami/nvme/models/MiMo-V2.5-text}"
BS="${BS:-32}"
SEQ="${SEQ:-1024}"
PORT="${PORT:-8000}"

# Mandatory on this instance: the Neuron runtime otherwise probes for an EFA
# device per rank and every rank dies with "No EFA device found at
# /sys/bus/pci/devices/.../infiniband".
export NEURON_SKIP_EFA_AFFINITY=1

echo "Starting MiMo-V2.5 server: $MODEL (BS=$BS, SEQ=$SEQ, TP=64, EP=64)"

vllm serve "$MODEL" \
    --served-model-name MiMo-V2.5 \
    --max-model-len "$SEQ" \
    --max-num-seqs "$BS" \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
    --additional-config "{
        \"neuron_config\": {
            \"quantization\": \"bf16\",
            \"ep_degree\": 64,
            \"num_batched_tokens_buckets\": [$SEQ],
            \"num_seqs_buckets\": [$BS],
            \"on_device_sampling_config\": {\"all_greedy\": \"true\"}
        }
    }" \
    --port "$PORT"
