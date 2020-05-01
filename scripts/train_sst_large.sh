#!/bin/bash

other_args=$@

#working_dir="/checkpoint/strubell/sledgehammer"
working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

#layers="0_3_5_11"
#bert_model="bert-base-uncased"
layers="0_4_12_23"
bert_model="bert-large-uncased"

#batch_size=72
# 64 doesn't work
batch_size=32

srun --gres=gpu:1 --constraint=volta32gb --time=12:00:00 python scripts/train_model.py \
-t $bert_model \
-l $layers \
--data_dir $data_dir \
-d $dataset \
-w $working_dir \
-b $batch_size \
$other_args