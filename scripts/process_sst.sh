#!/bin/bash

ORIG_DATA_DIR="/private/home/strubell/research/data/glue_data"
LOCAL_DATA_DIR="data_dir/text_cat"

DATASET="SST-2"
ORIG_DATASET_DIR="$ORIG_DATA_DIR/$DATASET"
LOCAL_DATASET_DIR="$LOCAL_DATA_DIR/$DATASET"

mkdir -p $LOCAL_DATASET_DIR

for split in train dev test; do
  f="$ORIG_DATASET_DIR/$split.tsv"
  echo "Processing: $f"
  tail -n +2 $f | awk '{label=$NF; $NF=""; print label"\t"$0}' > "$LOCAL_DATASET_DIR/$split"
done