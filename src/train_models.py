import numpy as np
import torch
import os
import argparse
import timeit
from datetime import datetime
import pickle
from pathlib import Path

import platform             # to get system information

from sGDML.sgdml import train
from sGDML.sgdml import cli
from tools import create_data, plot_data
from cluster_main import select_value


from typing import Dict, List, Union, TypeVar


def initialize_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=-1, help='iterator over all possible combinations.')
    # parser.add_argument('--repetitions', type=int, default=1, help='Run several repetitions for preconditioners.')

    # general input
    parser.add_argument('--name_dataset', default=['nanotube'], nargs='+',
                        choices=['aspirin', 'uracil', 'ethanol', 'azobenzene', 'toluene', 'catcher', 'nanotube'])
    parser.add_argument('--n_datapoints', default=5, type=int)
    parser.add_argument('--max_percentage', default=0.5, type=float, help='relative number of columns in percentage, '
                                                                        'largest preconditioner used')
    parser.add_argument('--absolut_path_to_script', type=str, default='./')
    parser.add_argument('--hardware', type=str, default='cpu', choices=['cpu', 'gpu'])

    # cg_steps
    # create_data.cg_steps(n_datapoints=25, n_measurements=5, max_percentage=1, min_columns=120, name_dataset='aspirin',
    #                      flag_eigvals=True, flags=flags)
    #  choices=['eigvec_precon', 'lev_random', 'cholesky', 'random_scores', 'lev_scores', 'inverse_lev',
    #  'truncated_cholesky_custom', 'rank_k_lev_scores']
    parser.add_argument('--preconditioner', type=str, nargs='+',
                        default=['random_scores']
                        # default=['truncated_cholesky', 'cholesky']
    )

    parser.add_argument('--calculate_eigvals', default=False, type=lambda x: (str(x).lower() == 'true'))

    # preconditioner_size
    # parser.add_argument('--list_n_datapoints', default=[50], nargs='+', type=int)
    # parser.add_argument('--preconditioner', type=str, default='eigvec_precon')

    # create_data.minimum_preconditioner_size(n_measurements=7, max_percentage=0.5, min_columns=500, name_dataset='uracil',
    #                                         flag_eigvals=False, list_n_datapoints=[750], datapoint_distr='linear')

    args = parser.parse_args()
    print('after convertion calculate_eigvals', args.calculate_eigvals)
    # flags = {'truncated_cholesky': False, 'eigvec_precon_atomic_interactions': False, 'eigvec_precon_block_diagonal': False,
    #          'eigvec_precon': False, 'lev_scores': False, 'random_scores': False, 'inverse_lev': False,
    #          'lev_random': False,  'cholesky': False}
    # len_flags = len(flags)
    # for str_preconditioner in args.list_preconditioner:
    #     if flags[str_preconditioner] is False:      # only access if key is already defined
    #         flags[str_preconditioner] = True        # activate preconditioner
    # assert len_flags == len(flags), 'Accidentally added a flag'
    # args.flags = flags
    return args


def train_model(index: int, name_dataset: str, solver: str, n_datapoints: int, preconditioner: str,
                path_to_script: str, hardware: str) -> Dict:
    assert hardware in ['cpu', 'gpu'], 'Only cpu or gpu allowed as options for hardware.'
    name_dataset, i_interal, n_options = select_value(values=name_dataset, index=index, n_options=1)
    n_datapoints = create_data.normalize_to_aspirin(n_datapoints=n_datapoints, dataset_name=name_dataset)

    preconditioner, i_interal, n_options = select_value(values=preconditioner, index=i_interal, n_options=n_options)

    print(f'dataset = {name_dataset} \nn_datapoints = {n_datapoints}')
    print(f'Running task {index} from a total of {n_options}.')
    assert n_options > index, 'All possible options have been exhausted.'

    # loop: analytic/iterative
    # train model
    if hardware == 'gpu':       # only run if gpu is really available
        assert torch.cuda.is_available(), f'Trying to train with GPU (hardware = {hardware}) without cuda device.'

    dataset = create_data.get_dataset(path_to_script=path_to_script, name_dataset=name_dataset)
    gdml_train = train.GDMLTrain(use_torch=True if hardware == 'gpu' else False, max_processes=1)
    task = gdml_train.create_task(dataset, int(n_datapoints),
                                  valid_dataset=dataset, n_valid=1000,
                                  sig=10, lam=1e-15, solver=solver)

    task['truncated_cholesky'] = 1500
    task['str_preconditioner'] = preconditioner

    n = task['F_train'].size
    m, k_min, _ = plot_data.get_params(dataset_name=name_dataset, old=False)
    k_RoT = int(plot_data.rule_of_thumb(n=n, k_min=k_min, m=m))
    preconditioner_strength = k_RoT / n

    def callback(*args, **kwargs):
        pass

    start_train = timeit.default_timer()
    model = gdml_train.train(task=task, break_percentage=preconditioner_strength, callback=callback,
                             str_preconditioner=preconditioner, flag_eigvals=False)
    stop_train = timeit.default_timer()
    model['solver_runtime_s'] = stop_train - start_train
    model['truncated_cholesky'] = 1500
    model['str_preconditioner'] = preconditioner
    model['n_datapoints'] = n_datapoints
    model['kernel_size'] = n
    model['preconditioner_strength'] = preconditioner_strength
    model['task'] = task
    model['hardware'] = hardware

    # store model and training time
    # rmse = cli.test(model_dir=model, test_dataset=dataset, n_test=100,
    #                 overwrite=False, max_processes=1, use_torch=False)
    # calculate error
    # sgdml test model.npz aspirin_dft.npz 100
    # sgdml test ../data_new/models/uracil/analytic/n\ =\ 35/2022-12-14_1655.npz uracil_dft.npz 100
    del gdml_train
    del task

    return model


def store_model(model: Dict):
    name_dataset = str(model['dataset_name'])
    n_datapoints = model['n_datapoints']
    preconditioner = model['str_preconditioner']

    now = datetime.now()
    # add key which indicates gpu: yes/no
    folder_name = Path(os.path.abspath(path_to_script)) / 'data_new' / 'models' / model['hardware'] / name_dataset

    solver = model['solver_name']
    file_name = f"{now.date()}_{now.strftime('%H%M')}.pickle"
    if solver == 'analytic':
        path = folder_name / 'analytic' / f'n={n_datapoints}' / file_name
    elif solver == 'cg':
        k = model['inducing_pts_idxs'].size
        path = folder_name / preconditioner / f'n={n_datapoints}' / f'k={k}' / file_name
    else:
        raise ValueError(f'solver = {solver}')

    uname = platform.uname()
    model['platform'] = uname

    path.parent.mkdir(exist_ok=True, parents=True)
    # with open(path, "wb") as file:
    #     pickle.dump(model, file)
    file_model_npz = path.with_suffix('.npz')
    print(f'Store model at {file_model_npz}')
    np.savez_compressed(file_model_npz, **model)


if __name__ == '__main__':
    args = initialize_parser()
    path_to_script = os.path.abspath(args.absolut_path_to_script)
    # index = args.index
    for index in range(100):
        for solver in ['analytic', 'cg']:
            index_internal = index if args.index == -1 else args.index
            model = train_model(index=index_internal, name_dataset=args.name_dataset, preconditioner=args.preconditioner,
                                solver=solver, n_datapoints=args.n_datapoints, path_to_script=path_to_script,
                                hardware=args.hardware)
            store_model(model=model)
        if args.index != -1:
            assert False, f'Script running with explicit index: {args.index}'

    # rsync -avz mlc:~/Projects/cholesky/project/data_new/ /Users/sbluecher/Documents/University/git/cholesky/project/data_new/ --include='*.pickle' --include='*.npz'
