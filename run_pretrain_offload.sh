#!/bin/bash

rm -f train_zero_bubble.log
rm -f gpu_log.csv


# Start GPU logging in background and record its PID
./log_gpu.sh &
LOG_PID=$!
# export OFFLOAD=1
export INTERLEAVED_1F1B=1
export INTERLEAVE_GROUP=8
export OFFLOAD_CHUNK_NUM=1

# Run training
./pretrain_offload.sh >> train_zero_bubble.log 2>&1

# After training finishes, kill the logging process
kill $LOG_PID

# Create next result_x folder
RESULT_BASE="data/zero_bubble_result"
mkdir -p "$RESULT_BASE"

# Find the next available index
LAST_INDEX=$(find "$RESULT_BASE" -maxdepth 1 -type d -name "result_*" \
             | sed 's/.*result_//' | sort -n | tail -n 1)

# If no result_x found, start from 1
if [[ -z "$LAST_INDEX" ]]; then
  NEXT_INDEX=1
else
  NEXT_INDEX=$((LAST_INDEX + 1))
fi

RESULT_DIR="${RESULT_BASE}/result_${NEXT_INDEX}"
mkdir -p "$RESULT_DIR"

# Copy result files
cp gpu_log.csv pretrain_offload.sh train_zero_bubble.log run_pretrain_offload.sh "$RESULT_DIR"

python data/zero_bubble_result/plot_memory.py  $RESULT_DIR
python data/zero_bubble_result/plot_multiple_info.py  $RESULT_DIR