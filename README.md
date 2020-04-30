# AI6127 DEEP NEURAL NETWORK FOR NATURAL LANGUAGE PROCESSING

## Set up environment

```shell script
conda env create -f environment.yml
```

You need to install pytorch, transformers and tensorboardX to run our code.

## Train and evaluate

0. albert baseline

```shell script
python main.py \
      --model_type albertbase \
      --model_name_or_path albert-base-v1 \
      --output_dir ./baseline \
      --data_dir /home/qmail/repos/nlp/data/ \
      --version_2_with_negative \
      --do_train \
      --do_eval \
      --num_train_epochs 3 \
      --per_gpu_train_batch_size 4 \
      --per_gpu_eval_batch_size 4
```

1. train

```shell script
python main.py \
      --model_type albert \
      --model_name_or_path albert-base-v1 \
      --output_dir ./exp01 \
      --data_dir /home/qmail/repos/nlp/data/ \
      --version_2_with_negative \
      --do_train \
      --num_train_epochs 3 \
      --per_gpu_train_batch_size 4
```

2. evaluate

```shell script
python main.py \
      --model_type albert \
      --model_name_or_path ./exp01 \
      --output_dir ./exp01 \
      --data_dir /home/qmail/repos/nlp/data/ \
      --version_2_with_negative \
      --do_eval \
      --num_train_epochs 3 \
      --per_gpu_eval_batch_size 4
```

## About our code

This code is cleaned up, you can check:

- `albert.py`: our model implementation with co-attention
- `args.py`: available arguments
- `main.py`: main script
