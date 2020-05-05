#!/bin/bash

other_args=$@

#working_dir="/checkpoint/strubell/sledgehammer"
working_dir="/private/home/strubell/research/sledgehammer/models"
data_dir="/private/home/strubell/research/sledgehammer/data_dir"
dataset="SST-2"

export EARLY_EXIT="true"
export SHARE_CLASSIFIERS="true"
export POOL_LAYERS="true"
margin="1.0"
export MARGIN=$margin
experiment_name="margin-$margin-early-exit-shared-everylayer"

#layers="0_3_5_11"
layers="0_1_2_3_4_5_6_7_8_9_10_11"
num_epochs=2

bert_model="bert-base-uncased"

config_file="training_config/sledgehammer_bert_classification_margin.jsonnet"

batch_size=72

#srun --gres=gpu:1 --constraint=volta32gb --time=12:00:00
python scripts/train_model.py \
-t $bert_model \
-l $layers \
--data_dir $data_dir \
-d $dataset \
-w "$working_dir/$experiment_name" \
-b $batch_size \
--training-config-file $config_file \
--num_epochs $num_epochs \
--margin $margin \
$other_args