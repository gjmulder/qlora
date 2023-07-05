#!/bin/sh

HF_USER=openlm-research
BASE_MODEL=open_llama_7b
DATA=oasst1
EPOCH=`date '+%s'`
RUN=`expr $EPOCH - 1677862104`
LORA_CHKPTS="$BASE_MODEL-$DATA-$RUN"

CUDA_VISIBLE_DEVICES=0 python qlora.py \
  --model_name_or_path $HF_USER/$BASE_MODEL \
  --dataset $DATA \
  --eval_dataset_size 2000 \
  --bits 8 \
  --evaluation_strategy steps \
  --logging_steps 20 \
  --eval_steps 20 \
  --save_steps 20 \
  --no_skip_memory_metrics \
  --output_dir /data/lora/finetuned_models/$LORA_CHKPTS
