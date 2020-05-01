#!/bin/bash

other_args=$@

#working_dir="/checkpoint/strubell/sledgehammer"
working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

layers="0_3_5_11"
bert_model="bert-base-uncased"

config_file="sledgehammer_bert_classification_margin.jsonnet"

batch_size=72

#srun --gres=gpu:1 --constraint=volta32gb --time=12:00:00
python scripts/train_model.py \
-t $bert_model \
-l $layers \
--data_dir $data_dir \
-d $dataset \
-w $working_dir \
-b $batch_size \
--training-config-file $config_file \
$other_args