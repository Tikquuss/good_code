#!/bin/bash

none="_None_"

### usage ###
# . train.sh $train_data_pct $math_operator $weight_decay $dropout $opt $max_lr $random_seed $use_wandb $group_name
# all this parameters are optional (see the default values below)

### params ###
train_data_pct=${1-5}
math_operator=${2-+}
weight_decay=${3-1}
dropout=${4-0.0}
opt=${5-adamw}
max_lr=${6-0.001}
random_seed=${7-0}

max_steps=100000
max_epochs=100000

### wandb ###
# wandb_entity is the name of the team on wandb and is optional
# wandb_project is the name of the project
use_wandb=False
#group_name="tdp=${train_data_pct}-wd=${weight_decay}-d=${dropout}-opt=${opt}-mlr=${max_lr}-rs=${random_seed}-mo${math_operator}"
# remove random_seed in group_name
group_name="tdp=${train_data_pct}-wd=${weight_decay}-d=${dropout}-opt=${opt}-mlr=${max_lr}-mo${math_operator}"
wandb_entity="grokking_ppsp"
wandb_project="grokking_operator=${math_operator}"

watch=$none
#watch="log=str(all),log_freq=int(1)"

### Experiment dump path ###
dump_path=..
logdir=${dump_path}/logs/$group_name
datadir=${dump_path}/data/$group_name

### Early_stopping (for grokking) : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold` ###
#early_stopping_grokking=$none
early_stopping_grokking="patience=int(1000),metric=str(val_accuracy),metric_threshold=float(90.0)"

###
python train.py \
		--batchsize -1 \
		--n_layers 2 \
		--n_heads 4 \
		--d_model 128 \
		--dropout $dropout \
		--weight_noise 0.0 \
		--non_linearity relu \
		--max_context_len 50 \
		--math_operator $math_operator \
		--train_data_pct $train_data_pct \
		--warmup_steps 10 \
		--anneal_lr_steps 100000 \
		--anneal_lr False \
		--max_lr $max_lr \
		--weight_decay $weight_decay \
		--weight_decay_kind to_zero \
		--noise_factor 0 \
		--save_activations False \
		--save_outputs False \
		--logdir $logdir \
		--datadir $datadir \
		--save_checkpoint True \
		--use_wandb $use_wandb \
		--group_name $group_name \
		--wandb_entity $wandb_entity \
		--wandb_project $wandb_project \
		--watch $watch \
		--opt $opt \
		--momentum 0.9 \
		--random_seed $random_seed \
		--max_steps $max_steps \
		--max_epochs $max_epochs \
		--accelerator auto \
		--devices auto \
		--early_stopping_grokking $early_stopping_grokking \
		--eval_only False \
#		--load_from_ckpt None \
#		--operand_length \
