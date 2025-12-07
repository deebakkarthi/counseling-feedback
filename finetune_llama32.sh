#!/bin/sh

mlx_lm.lora --model "meta-llama/Llama-3.2-1B-Instruct"\
        --train \
        --data "./data" \
        --batch-size 1\
        --num-layers 4
