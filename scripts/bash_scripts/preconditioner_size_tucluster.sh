#!/usr/bin/env bash
#$ -N pcon_nanotube       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -binding linear:8
#$-l h_vmem=20G
#$-l mem_free=20G
#$ -wd /home/bluecher31/Projects/cholesky/project       # change working directory (to current) # adapt me to your log folder
#$ -V          # provide environment variables
#$ -t 1-1
script_directory='/home/bluecher31/Projects/cholesky/project'
cd $script_directory
echo current working directory
pwd

echo 'I am the job with task ID TSGE_TASK_ID'
echo SGE_TASK_ID
echo $SGE_TASK_ID

nvidia-smi
it=$(($SGE_TASK_ID - 1))
model_iter=$((($it) % 2))
source

echo calling script with following command
~/miniconda3/envs/cholesky/bin/python $script_directory/cluster_main.py --name_routine 'preconditioner_size' --absolut_path_to_script $script_directory \
--list_n_datapoints 67 --name_dataset 'nanotube' --n_measurements 20 --datapoint_distr 'log' \
--min_columns 2000 --max_percentage 0.25 --preconditioner 'lev_random'
