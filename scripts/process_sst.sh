#!/bin/bash

ORIG_DATA_DIR="/private/home/strubell/research/data/glue_data/"
LOCAL_DATA_DIR="data_dir/text_cat"
DATASET="SST-2"

for split in "dev test"; do
  f="$SST_DIR/$split.tsv"
  awk '{label=$NF; $NF=""; print label"\t"$0}' $f > "$LOCAL_DATA_DIR/$DATASET/$split"
done