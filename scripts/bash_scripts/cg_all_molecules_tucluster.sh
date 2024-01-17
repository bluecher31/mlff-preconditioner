#!/usr/bin/env bash
#$ -N cg_all       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -binding linear:4
#$-l h_vmem=8G
#$-l mem_free=8G
#$ -wd /home/bluecher31/Projects/cholesky/project       # change working directory (to current) # adapt me to your log folder
#$ -V          # provide environment variables
#$ -t 1-7
script_directory='/home/bluecher31/Projects/cholesky/project'
cd $script_directory
echo current working directory
pwd

echo 'I am the job with task ID TSGE_TASK_ID'
echo $SGE_TASK_ID

nvidia-smi
molecule_dataset=("aspirin" "ethanol" "uracil" "azobenzene" "toluene" "nanotube" "catcher")
d_atoms=(21 9 12 24 15 370 88)

it=$(($SGE_TASK_ID - 1))
model_iter=$((($it) % 2))


~/miniconda3/envs/cholesky/bin/python $script_directory/cluster_main.py --name_routine 'cg_steps' --absolut_path_to_script $script_directory \
	--n_datapoints $((250*21/d_atoms[$it])) --name_dataset "${molecule_dataset[$it]}" --n_measurements 30 --max_percentage 0.25 --min_columns 150 \
 --list_preconditioner 'lev_random' 'cholesky' 'random_scores' 'eigvec_precon' 'lev_scores' 'inverse_lev' --datapoint_distr 'log'
