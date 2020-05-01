#!/bin/bash

#working_dir="/checkpoint/strubell/sledgehammer"
working_dir="/private/home/strubell/research/slegehammer/models"
data_dir="/private/home/strubell/research/slegehammer/data_dir"
dataset="SST-2"

bert_model="bert-base-uncased"

python scripts/train_model.py \
-t $bert_model \
-l 0_3_5_11 \
--data_dir $data_dir \
-d $dataset \
-w $working_dir