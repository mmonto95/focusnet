#!/bin/sh
#SBATCH --time=02-00:00:00
#SBATCH --job-name=fourier
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=mloper23@eafit.edu.co
#SBATCH --partition=accel-2
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --nodelist=compute-0-11
#SBATCH --ntasks-per-node=32

export PYTHONPATH="/home/mmonto95/focusnet:$PYTHONPATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python fft_train.py
