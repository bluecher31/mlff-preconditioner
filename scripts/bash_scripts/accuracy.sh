#!/usr/bin/env bash
#$ -N small_models       # set consistent base name for output and error file (allows for easy deletion alias)
#$-l h_vmem=100G
#$-l mem_free=100G
#$ -q all.q    # don't fill the qlogin queue
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -binding linear:1
#$ -wd /home/bluecher31/Projects/cholesky/project/bash_scripts       # change working directory (to current) # adapt me to your log folder
#$ -V          # provide environment variables
#$ -t 1-7
script_directory='/home/bluecher31/Projects/cholesky/project'
cd $script_directory
echo current working directory
pwd

echo 'I am the job with task ID TSGE_TASK_ID'
echo $SGE_TASK_ID

nvidia-smi
it=$(($SGE_TASK_ID - 1))


echo calling script with following command
~/miniconda3/envs/cholesky/bin/python $script_directory/train_models.py --index $it --absolut_path_to_script $script_directory \
 --n_datapoints 100 --name_dataset 'aspirin' 'uracil' 'ethanol' 'azobenzene' 'toluene' 'catcher' 'nanotube' --calculate_eigvals False\
 --preconditioner 'random_scores' --hardware 'gpu'
