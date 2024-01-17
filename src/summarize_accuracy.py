import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import itertools

from tools import plot_data, create_data

from typing import List, Dict


def extract_model_dict(npz_file) -> dict:
    """Converts all elements of npz_file into key: value and removes redundant arrays into correct (int, float,...)."""
    model = {}
    for key in npz_file.keys():
        value = npz_file.get(key)
        if value.size == 1:  # convert to int, float or str
            if np.issubdtype(value.dtype, dict):
                value = value.item()
            else:
                for class_type in [bool, int, float, str]:
                    if np.issubdtype(value.dtype, class_type):
                        value = class_type(value)
                        break
        model[key] = value
    return model


def test_model(path_to_script: str, path_model: Path, name_dataset: str):
    n_test = 100
    if name_dataset == 'larger_aims_nanotube' or name_dataset == 'nanotube':
        file_name_dataset = f'larger_aims_nanotube.npz'
    elif name_dataset == 'aims_catcher' or name_dataset == 'catcher':
        file_name_dataset = f'aims_catcher.npz'
    elif name_dataset == 'azobenzene_new' or name_dataset == 'azobenzene':
        file_name_dataset = f'azobenzene_dft.npz'
    elif name_dataset == 'C6H5CH3' or name_dataset == 'toluene':
        file_name_dataset = f'toluene_dft.npz'
    else:
        file_name_dataset = f'{name_dataset}_dft.npz'
    path_dataset = f'{path_to_script}/sGDML/{file_name_dataset}'
    os.system(f'sgdml test -p 1 {path_model} {path_dataset} {n_test}')


def load_model(base_path: str, solver: str, preconditioner: str, k: int, name_dataset: str, n_datapoints: int,
               hardware: str) -> List[Dict]:
    dataset_name = plot_data.map_dataset_name_to_molecule(name_dataset, reverse_order=True)
    folder_name = Path(os.path.abspath(base_path)) / 'data_new' / 'models' / hardware / dataset_name.lower()
    if solver == 'analytic':
        folder = folder_name / 'analytic' / f'n={n_datapoints}'
    elif solver == 'cg':
        folder = folder_name / preconditioner / f'n={n_datapoints}' / f'k={k}'
    else:
        raise ValueError(f'solver = {solver}')

    print(folder)
    paths = [path for path in folder.glob('*.npz')]

    assert len(paths) > 0, f'Could not find any model: folder path = {folder}\n' \
                           f'dataset = {name_dataset} \n' \
                           f'solver = {solver} \n' \
                           f'n_datapoints = {n_datapoints}\n' \
                           f'preconditioner = {preconditioner}\n'
    # if calculate_test_accuracy:
    models = []
    for path in paths:
        model_file = np.load(path, allow_pickle=True)
        model = extract_model_dict(npz_file=model_file)
        if np.isnan(model['e_err']['mae']):
            test_model(path_to_script=base_path, path_model=path, name_dataset=name_dataset)
            model_file = np.load(path, allow_pickle=True)
            model = extract_model_dict(npz_file=model_file)
        models.append(model)

    # model_files = [np.load(path, allow_pickle=True) for path in paths]
    # models = [extract_model_dict(npz_file=model_file) for model_file in model_files]
    return models


def calculate_average_scalar(models: List[Dict], key: str) -> [float, float]:
    """Average over scalar valued quantities in a model."""
    values = [model[key] for model in models]
    return np.mean(values), np.std(values)/np.sqrt(len(values))


def calculate_average_dict(models: List[Dict], key_target: str, measure: str) -> [float, float]:
    """
    Average over dictionary valued quantities in a model.

    Args:
        models: a list of trained model
        key_target: outer dictionary, ['f_err', 'e_err']
        measure: inner dictionary, ['mae', 'rmse']
    """
    values = [model[key_target][measure] for model in models]
    return np.mean(values), np.std(values)


def convert_to_table(df: pd.DataFrame, keys: List[str]) -> str:
    """Converts to a table which can be inserted into latex. (key +- error_key)"""

    # dicts: {molecule: {runtime: 1.23, }}
    for key in keys:
        print()
        mean = df[key]
        error = df[f'error_{key}']


if __name__ == '__main__':
    path_to_script = './'
    n_datapoints_raw = 1000
    hardware = 'gpu'

    df = pd.DataFrame()
    for name_dataset in [ 'uracil', 'ethanol', 'azobenzene', 'aspirin', 'toluene', 'catcher', 'nanotube']:
        n_datapoints = create_data.normalize_to_aspirin(n_datapoints=n_datapoints_raw, dataset_name=name_dataset)
        n = create_data.calculate_kernel_size(n_datapoints=n_datapoints, dataset_name=name_dataset)
        m, k_min, _ = plot_data.get_params(dataset_name=name_dataset, old=False)
        k_RoT = int(plot_data.rule_of_thumb(n=n, k_min=k_min, m=m))
        preconditioner_strength = k_RoT / n
        k = int(preconditioner_strength * n)

        dict_meas = {}
        n_measurements = None
        for solver in ['analytic', 'cg']:
            models = load_model(base_path=path_to_script, solver=solver, preconditioner='random_scores', k=k,
                                n_datapoints=n_datapoints, name_dataset=name_dataset, hardware=hardware)
            mean_runtime, error_runtime = calculate_average_scalar(models=models, key='solver_runtime_s')
            mean_accuracy, error_accuracy = calculate_average_dict(models=models, key_target='f_err', measure='mae')
            n_datatpoints, _ = calculate_average_scalar(models=models, key='n_datapoints')
            n_measurements = len(models) if n_measurements is None else min(n_measurements, len(models))

            dict_meas.update({f'runtime_{solver}': mean_runtime, f'accuracy_{solver}': mean_accuracy})
            dict_meas.update({f'error_runtime_{solver}': error_runtime, f'error_accuracy_{solver}': error_accuracy})
        dict_meas.update({'n_datapoints': int(n_datatpoints)})
        df_row = pd.Series(data=dict_meas, name=name_dataset)
        df = df.append(df_row)

    df = df.sort_values('n_datapoints', ascending=False)

    df['relative_speedup_cg'] = df['runtime_analytic'] / df['runtime_cg']
    df['n_measurements'] = n_measurements
    df['runtime_analytic_min'] = df['runtime_analytic']/60
    df['error_runtime_analytic_min'] = df['error_runtime_analytic'] / 60
    df['runtime_cg_min'] = df['runtime_cg'] / 60
    df['error_runtime_cg_min'] = df['error_runtime_cg'] / 60
    df['delta_accuracy'] = df['accuracy_analytic'] - df['accuracy_cg']
    # print(np.unique(df['n_datapoints']))
    # print(df.sort_index(axis=1))
    # table = convert_to_table(df=df, keys=['runtime_analytic_min', 'runtime_cg_min',
    #                                       'accuracy_analytic', 'accuracy_cg'])
    columns_nested = [[key, f'error_{key}'] for key in
                      ['accuracy_analytic', 'accuracy_cg', 'runtime_analytic_min', 'runtime_cg_min']]
    columns = list(itertools.chain(*columns_nested))


    print(df.to_latex(columns=columns, sparsify=True))
    print(df[['n_datapoints', 'runtime_analytic', 'runtime_cg', 'relative_speedup_cg']])

    print(df[['error_runtime_analytic', 'error_runtime_cg']])

    print(df[['n_measurements']])

    plt.figure('relative_speed_up', figsize=(7, 2))
    df['relative_speedup_cg'].plot.bar(color='C3', width=0.9)
    ax = plt.gca()
    ax.axhline(y=1, linewidth=3.5, color='grey')
    plt.ylabel('Speed ups')
    xticks = [t.capitalize() for t in df['relative_speedup_cg'].index]
    plt.xticks(ticks=ax.get_xticks(), labels=xticks, rotation=0)
    plt.tight_layout(pad=0.1)



