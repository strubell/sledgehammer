#!/bin/bash

ORIG_DATA_DIR="/private/home/strubell/research/data/glue_data/"
LOCAL_DATA_DIR="data_dir/text_cat"
DATASET="SST-2"
LOCAL_DATASET_DIR="$LOCAL_DATA_DIR/$DATASET"

mkdir -p $LOCAL_DATASET_DIR

for split in "dev test"; do
  f="$SST_DIR/$split.tsv"
  awk '{label=$NF; $NF=""; print label"\t"$0}' $f > "$LOCAL_DATASET_DIR/$split"
done