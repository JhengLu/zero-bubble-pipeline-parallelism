#!/bin/bash

ZERO_BUBBLE_SCRIPT="pretrain_zero_bubble.sh"

# Define values
micro_batch_sizes=(1 2 4 8)
global_batch_sizes=(8 12 16)

# Iterate over combinations
for micro in "${micro_batch_sizes[@]}"; do
  for global in "${global_batch_sizes[@]}"; do
    echo "Running with MICRO_BATCH_SIZE=$micro and GLOBAL_BATCH_SIZE=$global"

    # Replace values in pretrain_zero_bubble.sh
    sed -i "s/^[[:space:]]*MICRO_BATCH_SIZE=.*/  MICRO_BATCH_SIZE=$micro/" "$ZERO_BUBBLE_SCRIPT"
    sed -i "s/^[[:space:]]*GLOBAL_BATCH_SIZE=.*/  GLOBAL_BATCH_SIZE=$global/" "$ZERO_BUBBLE_SCRIPT"

    # Run the training scripts
    bash run_pretrain_original.sh
    bash run_pretrain_zero_bubble.sh
  done
done
