#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat. The repo's ./.cache is a
# symlink to it, so artifacts are reachable from inside the repo too.
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# # install uv (if not already installed)
# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# # create a .venv local virtual environment (if it doesn't exist)
# [ -d ".venv" ] || uv venv
# # install the repo dependencies
# uv sync --extra gpu
# # activate venv so that `python` uses the project's venv instead of system python

source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# # -----------------------------------------------------------------------------
# # During the course of the run, we will be writing markdown reports to the report/
# # directory in the base dir. This command clears it out and writes a header section
# # with a bunch of system info and a timestamp that marks the start of the run.
# python -m nanochat.report reset

# # -----------------------------------------------------------------------------
# # Tokenizer

# # Download the first ~2B characters of pretraining dataset
# # each data shard is ~250M chars
# # so we download 2e9 / 250e6 = 8 data shards at this point
# # each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# # look at dev/repackage_data_reference.py for details on how this data was prepared
# python -m nanochat.dataset -n 8
# # Immediately also kick off downloading more shards in the background while tokenizer trains
# # Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# # The maximum total number of shards available in the entire dataset is 6542.
# python -m nanochat.dataset -n 170 &
# DATASET_DOWNLOAD_PID=$!
# # train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
# python -m scripts.tok_train
# # evaluate the tokenizer (report compression ratio etc.)
# python -m scripts.tok_eval

# # -----------------------------------------------------------------------------
# # Base model (pretraining)
# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID

# # d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
# torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# # evaluate the model: CORE metric, BPB on train/val, and draw samples
# torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16



# device_batch_size: 单卡单次 fwd/bwd 处理的序列条数。
#   全局总 token 数 = device_batch_size × max_seq_len × world_size × grad_accum_steps
#   脚本会断言 total_batch_size 必须能被 (device_batch_size × max_seq_len × world_size) 整除。
#
# 原来 =16 报错的原因：
#   16 × 8192 × 8 = 1,048,576  > 自动算出的 total_batch_size = 524,288
#   单步 token 已是目标总 batch 的 2 倍，grad_accum 最小为 1 也无法整除 → AssertionError。
#
# 改成 =8 的原因 / 约束：
#   8 × 8192 × 8 = 524,288 = total_batch_size，正好整除，grad_accum_steps=1。
#   约束：device_batch_size × max_seq_len × 8 必须整除 524,288；
#         且受单卡显存上限限制（8192 长上下文下不能设太大）。
#         可选更小值 4（grad_accum_steps=2，更省显存但更慢）。

# torchrun --standalone --nproc_per_node=8 -m scripts_dev.exp2_base_train -- \
#     --depth=12 \
#     --window-pattern=L \
#     --target-param-data-ratio=40 \
#     --device-batch-size=8 \
#     --max-seq-len=8192 \
#     --run=d12_ctx8192 \
#     --no-smear \
#     --no-resid-lambdas \
#     --no-value-residual \
#     --no-backout \
#     2>&1 | tee /fsx/home/xudong.shen/work/attn-bias/nanochat/runs_dev/2_pretraining_d12_ctx8192.log


torchrun --standalone --nproc_per_node=8 -m scripts_dev.exp3_base_train -- \
    --depth=12 \
    --window-pattern=L \
    --target-param-data-ratio=20 \
    --device-batch-size=32 \
    --max-seq-len=2048 \
    --run=pretrain_d12_ctx2048 \
    --model-tag=pretrain_d12_ctx2048 \
    --save-every=1000 \
    --no-smear \
    --no-resid-lambdas \
    --no-value-residual \
    --no-backout \
    --no-rope \
    --no-qknorm \
    --no-qk-scale \
    2>&1 | tee /fsx/home/xudong.shen/work/attn-bias/nanochat/runs_dev/3_pretraining_d12_ctx2048.log



# # -----------------------------------------------------------------------------
# # SFT (teach the model conversation special tokens, tool use, multiple choice)

# # download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# # see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# # run SFT and eval the model
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# # chat with the model over CLI! Leave out the -p to chat interactively
# # python -m scripts.chat_cli -p "Why is the sky blue?"

# # even better, chat with your model over a pretty WebUI ChatGPT style
# # python -m scripts.chat_web

# # -----------------------------------------------------------------------------
# # Generate the full report by putting together all the sections
# # report.md is the output and will be copied to current directory for convenience
# python -m nanochat.report generate