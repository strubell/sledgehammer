#!/bin/bash

other_args=$@

working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

bert_model="bert-base-uncased"
layers="0_3_5_11"

experiment="baseline"
model="$working_dir/$experiment/$bert_model/$dataset/experiment_${layers}_0/best.th"

python scripts/run_calibration.py \
-m $model \
-v "$data_dir/text_cat/$dataset/dev" \
$other_args