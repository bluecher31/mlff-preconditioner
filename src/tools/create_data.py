import os.path
import os

import numpy as np

from . import custom_cg_solver as custom_cg
from . import utils
from sGDML.sgdml import train

from datetime import datetime
import pickle
import platform             # to get system information

from pathlib import Path

from typing import List, Dict


lam = 1E-9
info_keys = ['dataset_name', 'sig', 'lam', 'solver_tol']


def get_dataset(path_to_script: str, name_dataset: str):
    available_datasets = ['aspirin', 'ethanol', 'paracetamol', 'benzene', 'uracil', 'azobenzene', 'toluene']
    if path_to_script.startswith('.') is True:       # convert to absolute path
        path_to_script = os.path.abspath(path_to_script)
    path_to_datafolder = os.path.join(path_to_script, 'sGDML/')
    if available_datasets.__contains__(name_dataset):
        dataset = np.load(f'{path_to_datafolder}{name_dataset}_dft.npz')
    elif name_dataset == 'nanotube':
        dataset = np.load(f'{path_to_datafolder}larger_aims_{name_dataset}.npz')
    elif name_dataset == 'catcher':
        dataset = np.load(f'{path_to_datafolder}aims_{name_dataset}.npz')
    else:
        assert False, f'incorrect input dataset: {name_dataset}'
    return dataset


def convert_dataset_name(dataset: str) -> str:
    """Convert to internal name defined by sGDML."""
    if dataset == 'catcher':
        dataset = 'aims_catcher'
    elif dataset == 'nanotube':
        dataset = 'larger_aims_nanotube'
    elif dataset == 'azobenzene':
        dataset = 'azobenzene_new'
    elif dataset == 'toluene':
        dataset = 'C6H5CH3'
    return dataset


def get_number_of_atoms(dataset_name: str) -> int:
    if dataset_name == 'aspirin':
        d = 21
    elif dataset_name == 'ethanol':
        d = 9
    elif dataset_name == 'uracil' or dataset_name == 'benzene':
        d = 12
    elif dataset_name == 'toluene':
        d = 15
    elif dataset_name == 'azobenzene' or dataset_name == 'azobenzene_new':
        d = 24      # correct: 24, OLD to be a mistake: 26
    elif dataset_name == 'catcher' or dataset_name == 'aims_catcher':
        d = 88
    elif dataset_name == 'nanotube' or dataset_name == 'larger_aims_nanotube':
        d = 370
    else:
        raise NotImplementedError(f'dataset_name = {dataset_name} is not specified. ')
    return d


def normalize_to_aspirin(n_datapoints: int, dataset_name: str) -> int:
    """Rescale n_datapoints per basis of aspirin"""
    d = get_number_of_atoms(dataset_name=dataset_name)
    n_datapoints *= 21 / d
    return max(int(n_datapoints), 2)


def calculate_kernel_size(n_datapoints: int, dataset_name: str) -> int:
    """Calculates the kernel size based on the number of training points."""
    d = get_number_of_atoms(dataset_name=dataset_name)
    return 3 * d * n_datapoints


def create_task(n_datapoints: int, path_to_script='', name_dataset='aspirin') -> [Dict, train.GDMLTrain]:
    print('start create task')
    dataset = get_dataset(path_to_script=path_to_script, name_dataset=name_dataset)
    solver = 'cg'

    gdml_train = train.GDMLTrain(use_torch=True)
    task = gdml_train.create_task(dataset, int(n_datapoints),
                                  valid_dataset=dataset, n_valid=1000,
                                  sig=10, lam=1e-15, solver=solver)
    return task, gdml_train


def cg_steps(task: Dict, gdml_train: train.GDMLTrain, n_datapoints: int, preconditioner_strength: float, preconditioner: str, flag_eigvals=False,
             path_to_script=''):
    print('START cg_steps')
    name_dataset = str(task['dataset_name'])
    task['truncated_cholesky'] = 1500
    task['str_preconditioner'] = preconditioner

    def callback(*args, **kwargs):
        pass

    dic_cg_steps = {}
    # set all measurements to False as default

    print(f'preconditioner_strength = {preconditioner_strength:.1%}')
    model = gdml_train.train(task=task, break_percentage=preconditioner_strength, callback=callback,
                             str_preconditioner=preconditioner, flag_eigvals=flag_eigvals)

    actual_preconditioner_size = len(model['inducing_pts_idxs'])/len(model['alphas_F'])
    print(f'actual_preconditioner_strength = {actual_preconditioner_size:.1%}')
    n = len(model['alphas_F'])
    k = int(actual_preconditioner_size * n)
    print(f'k = {k}')

    if preconditioner == 'cholesky':
        t = model['time_cholesky']
        t_begin = np.median(t[:20])
        t_end = np.median(t[20:])
        correction = t_end/t_begin - 1
        dic_cg_steps['t_cholesky'] = t
        dic_cg_steps['time_cg_step'] = model['total_time_cg']/model['solver_iters']
        dic_cg_steps['chol_t_begin'] = t_begin
        dic_cg_steps['chol_t_end'] = t_end
        dic_cg_steps['chol_t_correction'] = correction

    if flag_eigvals is True:
        dic_cg_steps[f'eigvals_{preconditioner}_{preconditioner_strength*100:.2f}'] = model['eigvals']
        dic_cg_steps[f'eigvals_{preconditioner}_{0}'] = model['eigvals_K']

    if model['is_conv'] is False and flag_eigvals is False:     # stop run but not for eigvals measurements
        raise RuntimeError('Solver is not converged.')

    dic_cg_steps[f'{preconditioner}_percentage'] = actual_preconditioner_size
    dic_cg_steps[f'{preconditioner}_cgsteps'] = model['solver_iters']
    dic_cg_steps[f'K.shape'] = (len(model['alphas_F']), len(model['alphas_F']))
    dic_cg_steps[f'n_kernel'] = n
    dic_cg_steps['k'] = k
    dic_cg_steps['total_time_preconditioner'] = model['total_time_preconditioner']
    dic_cg_steps['total_time_solve'] = model['total_time_solve']
    dic_cg_steps['total_time_cg'] = model['total_time_cg']

    dic_cg_steps['task'] = task
    for label in info_keys:
        dic_cg_steps[label] = task[label]

    uname = platform.uname()
    dic_cg_steps['platform'] = uname

    dic_cg_steps['n_datapoints'] = n_datapoints
    now = datetime.now()
    folder_name = Path(os.path.abspath(path_to_script)) / 'data_new' / name_dataset / preconditioner \
        / f'n = {n_datapoints}'

    file_name = f"{now.date()}_{now.strftime('%H%M')}_k = {k}"
    if flag_eigvals is True:
        file_name += '_eigvals'
    file_name += '.pickle'

    folder_name.mkdir(exist_ok=True, parents=True)
    with open(folder_name / file_name, "wb") as file:
        pickle.dump(dic_cg_steps, file)
    print('FINISH cg_steps. \n\n')


def eigvals(n_datapoints: int, list_percentage: List[float]):
    """calculates eigvals spectrum for different precon levels"""
    # n_datapoints = 40
    # list_percentage = [0.6, 0.7]
    x_list = []
    y_list = []
    labels = []

    K, y = utils.get_sGDML_kernel_mat(n_train=n_datapoints)
    K_hat = K + lam * np.eye(K.shape[0])

    eigvals_K_hat = np.linalg.eigvals(K_hat)
    eig = np.abs(eigvals_K_hat)
    y_list.append(eig)
    x_list.append(np.linspace(0, 1, len(eig)))
    labels.append(f'K_hat: {K_hat.shape}')

    eigvals_K = np.linalg.eigvals(K)
    eig = np.abs(eigvals_K)
    y_list.append(eig)
    x_list.append(np.linspace(0, 1, len(eig)))
    labels.append(f'K')

    for break_percentage in list_percentage:
        M = custom_cg.init_percond_operator(K, lam, break_percentage)
        eigvals_P_K = np.linalg.eigvals(M @ K_hat)
        eig = np.abs(eigvals_P_K)
        y_list.append(eig)
        x_list.append(np.linspace(0, 1, len(eig)))
        labels.append(f'precon: {break_percentage}%')
    return x_list, y_list, labels


def minimum_preconditioner_size(n_measurements: int, max_percentage: float, min_columns: float, name_dataset='aspirin',
                                flag_eigvals=False, list_n_datapoints=[50, 75, 100, 150], str_preconditioner='lev_random', datapoint_distr='linear',
                                path_to_script=''):
    """
    This conduct experiments to calculate the minimum value of k. Therefore we run multiple CG runs for the same molecule
    but different number of training points. There one can check for which k a certain sublinear performance is reached.
    """
    print('START minimum_preconditioner_size')
    dataset = get_dataset(path_to_script=path_to_script, name_dataset=name_dataset)
    solver = 'cg'

    gdml_train = train.GDMLTrain(use_torch=True)

    dic_cg_steps = {}
    dic_cg_steps['list_n_datapoints'] = list_n_datapoints
    dic_cg_steps['preconditioner'] = str_preconditioner

    for n_datapoints in list_n_datapoints:
        print('start create_task')
        task = gdml_train.create_task(dataset, n_datapoints,
                                      valid_dataset=dataset, n_valid=1000,
                                      sig=10, lam=1e-15, solver=solver)

        n_columns = task['F_train'].size        # number of columns in K, i.e. K.shape = (n_columns, n_columns)
        min_percentage = min_columns / n_columns
        if datapoint_distr == 'linear':
            list_percentage = np.linspace(max_percentage, min_percentage, n_measurements)       # begin with strongest precond.
        elif datapoint_distr == 'log':
            list_percentage = np.geomspace(max_percentage, min_percentage, n_measurements)  # begin with strongest precond.
        else:
            assert False, f'incorrect keyword: {datapoint_distr}'

        def callback(*args, **kwargs):
            pass

        # set all measurements to False as default
        flags = {'lev_scores': False, 'cholesky': False, 'random_scores': False, 'inverse_lev': False,  'lev_random': False,
                 'eigvec_precon': False}
        for i, key_preconditioner in enumerate([str_preconditioner]):     # include keys here to measure
            x_temp = []
            y_temp = []

            total_time_solve = []
            total_time_cg = []
            total_time_preconditioner = []
            for i, entry in enumerate(list_percentage):
                print(f'Evaluating entry {entry} number {i+1} out of {len(list_percentage)}')
                model = gdml_train.train(task=task, break_percentage=entry, callback=callback,
                                         str_preconditioner=key_preconditioner, flag_eigvals=flag_eigvals)

                x_temp.append(len(model['inducing_pts_idxs'])/len(model['alphas_F']))
                y_temp.append(model['solver_iters'])
                total_time_preconditioner.append(model['total_time_preconditioner'])
                total_time_solve.append(model['total_time_solve'])
                total_time_cg.append(model['total_time_cg'])

                if model['is_conv'] is False:
                    break

            dic_cg_steps[f'{n_datapoints}_{key_preconditioner}_percentage'] = np.array(x_temp)
            dic_cg_steps[f'{n_datapoints}_{key_preconditioner}_cgsteps'] = np.array(y_temp)
            dic_cg_steps[f'{n_datapoints}_K.shape'] = (len(model['alphas_F']), len(model['alphas_F']))
            dic_cg_steps[f'{n_datapoints}_{key_preconditioner}_total_time_preconditioner'] = \
                np.stack(total_time_preconditioner)
            dic_cg_steps[f'{n_datapoints}_{key_preconditioner}_total_time_cg'] = \
                np.stack(total_time_cg)
            dic_cg_steps[f'{n_datapoints}_{key_preconditioner}_total_time_solve'] = \
                np.stack(total_time_solve)


        for key_preconditioner in info_keys:
            dic_cg_steps[key_preconditioner] = task[key_preconditioner]

        # dic_cg_steps['n_datapoints'] = n_datapoints

    uname = platform.uname()
    dic_cg_steps['platform'] = uname
    now = datetime.now()
    file_name = f"{now.year}{now.month}{now.day}_{now.strftime('%H%M')}_" \
                f"precon_size_{name_dataset}_min{min(list_n_datapoints)}_max{max(list_n_datapoints)}"
    if flag_eigvals is True:
        file_name += '_eigvals'
    pickle.dump(dic_cg_steps, open(file_name, "wb"))

