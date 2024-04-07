#!/bin/bash

# Specify the GPU devices you want to use (replace 0,1 with the actual GPU indices)
export CUDA_VISIBLE_DEVICES=7

# Add arguments here as variable assignments (e.g., DATASET='mnist')
EXPERIMENT_TYPE='compression+filtering'
NUM_ENC_EPOCHS=10
MINE_BATCH_SIZE=-1
MINE_EPOCHS_UTILITY=2000
MINE_EPOCHS_PRIVACY=5000
BETA=5
DATASET='sst2'

# Define the encoder hidden sizes as an array of integers
ENCODER_HIDDEN_SIZES="512 256 128" 

COMBINATION_TYPE="premise_only"

# Run the Python script and pass the command line arguments
python train_encoder.py \
--experiment_type "$EXPERIMENT_TYPE" \
--save_enc_weights \
--num_enc_epochs "$NUM_ENC_EPOCHS" \
--mine_batch_size "$MINE_BATCH_SIZE" \
--mine_epochs_privacy "$MINE_EPOCHS_PRIVACY" \
--mine_epochs_utility "$MINE_EPOCHS_UTILITY" \
--beta "$BETA" \
--dataset "$DATASET" \
--encoder_hidden_sizes "$ENCODER_HIDDEN_SIZES" \
--combination_type "$COMBINATION_TYPE" \