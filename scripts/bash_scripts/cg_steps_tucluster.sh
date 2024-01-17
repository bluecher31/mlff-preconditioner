#!/usr/bin/env bash
#$ -N cg_catcher       # set consistent base name for output and error file (allows for easy deletion alias)
#$-l h_vmem=64G
#$-l mem_free=64G
#$ -q all.q    # don't fill the qlogin queue
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -binding linear:1
#$-l gputype='!(P100G12|GTX1080)'	# exclude small GPUs
#$ -wd /home/bluecher31/Projects/cholesky/project       # change working directory (to current) # adapt me to your log folder
#$ -V          # provide environment variables
#$ -t 1-1
script_directory='/home/bluecher31/Projects/cholesky/project'
cd $script_directory
echo current working directory
pwd

echo 'I am the job with task ID TSGE_TASK_ID'
echo $SGE_TASK_ID

nvidia-smi
it=$(($SGE_TASK_50ID - 1))
model_iter=$((($it) % 2))


echo calling script with following command
~/miniconda3/envs/cholesky/bin/python $script_directory/cluster_main.py --name_routine 'cg_steps' --absolut_path_to_script $script_directory \
 --n_datapoints 250 --name_dataset 'catcher' --n_measurements 12 --max_percentage 0.15 --min_columns 1000  --calculate_eigvals False \
 --list_preconditioner 'cholesky' 'random_scores' 'truncated_cholesky' 'lev_random' --datapoint_distr 'log' 
