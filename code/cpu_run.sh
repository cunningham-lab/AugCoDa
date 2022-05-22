#!/bin/sh
#
#SBATCH -A stats # The account name for the job.
#SBATCH --job-name=topSecret # The job name.
##SBATCH -c 4 # The number of cpu cores to use.
##SBATCH --exclusive # run on single node
#SBATCH --time=10:00:00 # The time the job will take to run.
##SBATCH --mem-per-cpu=16gb # The memory the job will use per cpu core.
##SBATCH --gres=gpu:1
##SBATCH -o log.log
##SBATCH --exclude=t118

source /moto/home/eg2912/.bashrc
conda activate sscoda

sleep 1

python $1
