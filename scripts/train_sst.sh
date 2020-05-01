#!/bin/bash

other_args=$@

#working_dir="/checkpoint/strubell/sledgehammer"
working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

layers="0_3_5_11"
bert_model="bert-base-uncased"

python scripts/train_model.py \
-t $bert_model \
-l $layers \
--data_dir $data_dir \
-d $dataset \
-w $working_dir \
$other_args