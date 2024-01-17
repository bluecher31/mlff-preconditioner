#!/usr/bin/env bash
#$ -N run_cpu_nano_small       # set consistent base name for output and error file (allows for easy deletion alias)
#$-l h_vmem=16G
#$-l mem_free=16G
#$ -q all.q    # don't fill the qlogin queue
#$ -l cuda=0   # request one GPU (remove this line if none is needed)
#$ -binding linear:1
#$ -wd /home/bluecher31/Projects/cholesky/project/bash_scripts       # change working directory (to current) # adapt me to your log folder
#$ -l h_rt=48:00:00
#$ -V          # provide environment variables
#$ -t 1-5
script_directory='/home/bluecher31/Projects/cholesky/project'
cd $script_directory
echo current working directory
pwd

echo 'I am the job with task ID TSGE_TASK_ID'
echo $SGE_TASK_ID

nvidia-smi
it=$(($SGE_TASK_ID - 1))


echo calling script with following command
~/miniconda3/envs/cholesky/bin/python $script_directory/cluster_main.py --index $it --absolut_path_to_script $script_directory \
 --n_datapoints 14 --name_dataset 'nanotube' --n_measurements 5 --max_percentage 0.0075 --min_columns 10  --calculate_eigvals False\
 --preconditioner 'eigvec_precon' --datapoint_distr 'log'
