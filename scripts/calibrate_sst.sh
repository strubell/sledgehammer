#!/bin/bash

other_args=$@

working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

bert_model="bert-base-uncased"
layers="0_3_5_11"

python scripts/run_calibration.py \
-m "$working_dir/$bert_model/$dataset/experiment_$layers/best.th" \
-v "$data_dir/text_cat/$dataset/dev" \
$other_args