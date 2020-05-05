#!/bin/bash

other_args=$@

working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

bert_model="bert-base-uncased"
layers="0_3_5_11"

experiment="baseline"
model_dir="$working_dir/$experiment/$bert_model/$dataset/experiment_${layers}_0"

temperatures="1.544136643409729_1.584019660949707_1.6111353635787964_1.582806944847107"

for confidence in $( seq 55 5 100 ); do
  python scripts/run_evaluation.py  \
  -t $temperatures \
  -c $( bc -l <<< "$confidence / 100" ) \
  -v "$data_dir/text_cat/$dataset/dev" \
  -o "$model_dir/eval" \
  -m "$model_dir/best.th" \
  $other_args
done
