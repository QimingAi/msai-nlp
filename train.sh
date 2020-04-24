#!/usr/bin/env bash

python main.py \
      --model_type albert \
      --model_name_or_path albert-base-v1 \
      --output_dir ./exp07_coattnx6_albert \
      --data_dir /home/qmail/repos/nlp/data/ \
      --version_2_with_negative \
      --do_train \
      --do_eval \
      --num_train_epochs 3 \
      --per_gpu_train_batch_size 4 \
      --per_gpu_eval_batch_size 4