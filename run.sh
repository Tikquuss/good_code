#!/bin/bash

for train_data_pct in 5 25 50 80; do {
for math_operator in +; do {
for weight_decay in 0 1; do {
for dropout in 0.0 0.1; do {
for opt in adamw sgd; do {
for max_lr in 0.001 0.01; do {
for random_seed in 0 100 500; do {
. train.sh $train_data_pct $math_operator $weight_decay $dropout $opt $max_lr $random_seed
} done
} done
} done
} done
} done
} done
} done