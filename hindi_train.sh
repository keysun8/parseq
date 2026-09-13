#!/bin/bash
#SBATCH -A ajoy.mondal
#SBATCH --nodelist=gnode074
#SBATCH -c 38
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=1024
#SBATCH --time=240:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

source activate parseq
nvidia-smi

cd /scratch/ajoy/printed/hindi/parseq/

python3 train.py +experiment=parseq trainer.max_epochs=10 model.batch_size=128 trainer.accelerator=gpu trainer.devices=4

#cd ~ 
#scp -r /scratch/ajoy/printed/hindi/parseq/outputs  0_new_experiments/3_parseq_printed/hindi/out/
