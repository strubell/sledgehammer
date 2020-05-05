#!/bin/bash

other_args=$@

working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

bert_model="bert-base-uncased"
layers="0_3_5_11"
margin="5.0"

#model_dir="$working_dir/$bert_model/$dataset/experiment_${layers}_0"
experiment="margin-$margin-early-exit-shared"
#experiment="margin-early-exit-shared"
model_dir="$working_dir/$experiment/$bert_model/$dataset/experiment_${layers}_0"

# dummy temps
temperatures="1_1_1_1"

conf_start=$( bc -l <<< "$margin * 50" )
conf_end=$( bc -l <<< "$margin * 200" )
conf_by=$( bc -l <<< "$margin * 5" )

for confidence in $( seq $conf_start $conf_by $conf_end ); do
#for confidence in $( seq 55 5 100 ); do
  python scripts/run_evaluation.py  \
  -t $temperatures \
  -c $( bc -l <<< "$confidence / 100" ) \
  -v "$data_dir/text_cat/$dataset/dev" \
  -o "$model_dir/eval" \
  -m "$model_dir/best.th" \
  $other_args
done
