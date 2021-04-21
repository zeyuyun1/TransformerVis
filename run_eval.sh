#!/bin/sh

for l in 0 2 4 6 8 10; do python inference.py --dictionary_dir ./dictionaries/bert-base-uncased_reg0.23_d2000_epoch2.npy --data_dir ./data/sentences_short.npy --reg 0.23 --num_instances 10000 --shard_size 1000 --l $l --gpu_id 1; done