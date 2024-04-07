#!/bin/bash

# Specify the GPU devices you want to use (replace 0,1 with the actual GPU indices)
export CUDA_VISIBLE_DEVICES=7

python calculate_mi.py