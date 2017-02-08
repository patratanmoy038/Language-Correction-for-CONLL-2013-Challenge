#!/bin/bash
#PBS -N tpatra_s2s
#PBS -P chemical
#PBS -l select=1:ncpus=1:ngpus=2
#PBS -l walltime=05:00:00

cd /home/chemical/dual/ch7130186/ELL881/Assignment_2
module load apps/tensorflow/0.11/gnu

python seq2seq_tf.py > logs/m300_auto.txt
