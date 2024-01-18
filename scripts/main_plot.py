import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import pickle
import glob

from src.tools import plot_data, create_data
from src.tools import init_plt as c_plt

from typing import Dict, List
from pathlib import Path

from operator import itemgetter


def memory_demand_matrix(n_kernel: int, k_preconditioner: int):
    matrix = np.zeros(shape=(n_kernel, k_preconditioner))
    size_MB = matrix.size * matrix.itemsize * 2 ** -20
    return size_MB


def check_compatibility(base_dict: Dict, list_dicts: List[Dict]):
    for d in list_dicts:
        for key in ['sig', 'lam', 'K.shape', 'dataset_name', 'n_datapoints']:
            assert base_dict[key] == d[key], f'Incompatibility for key = {key} found. \n' \
                                             f'base_dict[key] = {base_dict[key]}\n' \
                                             f'd[key] = {d[key]}\n'


def load_data_new(dataset: str, preconditioners: List[str], n_train: int, extract_key: str, normalize_n_train: bool) -> Dict:
    """
    This functions ensures compatibility to the previous format (everything in a single dict) by merging all subdicts.
    """
    if normalize_n_train:
        n_train = create_data.normalize_to_aspirin(n_datapoints=n_train, dataset_name=dataset)

    dict_data = None
    for preconditioner in preconditioners:
        dataset_name = create_data.convert_dataset_name(dataset)
        dir_path = Path('data_new').absolute() / dataset_name / preconditioner / f'n = {n_train}'
        assert dir_path.exists(), f'Could not find folder associated with the params.\n' \
                                  f'{dir_path}'
        print(f'Load files from directory: \n{dir_path}')
        files = [file for file in dir_path.glob('*.pickle')]
        list_dict_k = [pickle.load(open(file, 'rb')) for file in files]

        list_dict_k_sorted = sorted(list_dict_k, key=itemgetter('k'))
        if dict_data is None:
            dict_data = list_dict_k_sorted[0]
        else:
            check_compatibility(base_dict=dict_data, list_dicts=list_dict_k)
        if extract_key == 'cg_steps':
            np_percentage = np.stack([d[f'{preconditioner}_percentage'] for d in list_dict_k_sorted])
            np_cg_steps = np.stack([d[f'{preconditioner}_cgsteps'] for d in list_dict_k_sorted])
            dict_data[f'{preconditioner}_percentage'] = np_percentage
            dict_data[f'{preconditioner}_cgsteps'] = np_cg_steps
        else:
            raise ValueError(f'Cannot extract the requested keys. \n'
                             f'extract_key = {extract_key}')

    return dict_data


if __name__ == '__main__':
    c_plt.update_rcParams(half_size_image=False)
    # plot_data.visualize_rule_of_thumb()
    folder_data = '../data/'
    path_data = 'data/cg_performance_n=15750/2022-03-17_2333_ethanol_points583_meas31'
    # path_data = 'data/cg_truncated_cholesky/2022-03-17_2333_ethanol_points583_meas31'
    dic_data = pickle.load(open(folder_data + path_data, 'rb'))
    # plot_data.time_cholesky()
    # plot_data.cg_steps(dic_data, plot_save=False, title='old')

    list_dict_new = []
    # for molecule in ['aspirin']:
    # for molecule in ['aspirin', 'uracil', 'ethanol', 'azobenzene', 'toluene', 'catcher', 'nanotube']:
    # for molecule in ['aspirin', 'ethanol', 'catcher', 'nanotube']:
    #     dict_data_new = load_data_new(dataset=molecule, n_train=250,
    #                                   preconditioners=['eigvec_precon', 'lev_random', 'cholesky', 'random_scores'],
    #                                   # preconditioners=['rank_k_lev_scores', 'lev_random', 'random_scores'],
    #                                   extract_key='cg_steps', normalize_n_train=True)
    #     list_dict_new.append(dict_data_new)
        # plot_data.cg_steps(dict_data_new, title='new - ')
        # plot_data.cg_steps_difference(dict_data=dict_data_new, reference_label='eigvec_precon')
    # plot_data.cg_steps_difference_all(list_dict_data=list_dict_new, reference_label='eigvec_precon')
    # plot_data.cg_performance_multiple_molecules(list_data_dict=list_dict_new, selection='new', max_percentage=0.149)

    # compare methods for multiple molecule
    path_data = 'data/cg_performance_n=15750/2022-03-17_0905_aspirin_points250_meas31'
    dict_data = pickle.load(open(folder_data + path_data, 'rb'))
    plot_data.cg_steps(dic_data, plot_save=False, title='det_lev_scores')

    # plot_data.time_cholesky()

    # all_eigvals_path = glob.glob('data/eigenvalues/2022-*meas*_eigvals')
    # for path_eigvals in all_eigvals_path:
    #     dict_eigvals = pickle.load(open(path_eigvals, 'rb'))
    #     plot_data.eigvals(dict_eigvals, all_labels=['eigvec_precon', 'lev_random', 'cholesky', 'random_scores'],
    #                       n_eigvals=150)
    path_eigvals_aspirin = glob.glob(folder_data + 'data/eigenvalues/2022-*aspirin*meas*_eigvals')[0]
    plot_data.eigvals_schematic(dict_data=pickle.load(open(path_eigvals_aspirin, 'rb')))

    path_eigvals = 'data/eigenvalues/2022-07-12_1124_aspirin_points25_meas5_eigvals_jacobi'
    # path_eigvals = '2022-09-08_1505_catcher_points13_meas3_eigvals'
    dict_eigvals = pickle.load(open(folder_data + path_eigvals, 'rb'))
    plot_data.eigvals(dict_eigvals)

    path_ridge_levscores = 'data/ridge_leverage_scores/ridge_levscores_aspirin_15750'
    dict_ridge = pickle.load(open(folder_data + path_ridge_levscores, 'rb'))
    # plot_data.visualize_ridge_leverage_scores(dict_ridge)

    list_data = glob.glob(pathname=folder_data + 'data/cg_performance_n=15750_detailed/2022*')        # use for paper selection
    # list_data = glob.glob(pathname='data/cg_performance_n=15750/2022*')               # use for all molecules
    list_data_dict = [pickle.load(open(p, 'rb')) for p in list_data]
    plot_data.cg_performance_multiple_molecules(list_data_dict=list_data_dict, selection='paper_old')

    path_data = 'data/preconditioner_size/2021-12-23_0008_aspirin_points1000_meas5'
    dict_precon_aspirin = pickle.load(open(folder_data + path_data, 'rb'))
    plot_data.preconditioner_size(dict_data=dict_precon_aspirin)

    list_molecules_preoconditioner_size = glob.glob(folder_data + 'data/rule_of_thumb/estimate_slope, nmax = 31 500/*')
    plot_data.preconditioner_size_molecules(list_dict_molecules=[pickle.load(open(path_file, 'rb'))
                                                                 for path_file in list_molecules_preoconditioner_size])


    path_file = 'data/rule_of_thumb/estimate_slope, nmax = 31 500/2022310_1407_precon_size_aspirin_min100_max500'
    dict_train = pickle.load(open(folder_data + path_file, 'rb'))

    path_file = 'data/202235_0853_precon_size_aspirin_min1000_max1000'
    dict_test = pickle.load(open(folder_data + path_file, 'rb'))

    # all_molecules_filepath = glob.glob('data/rule_of_thumb/estimate_slope, nmax = 31 500/2022*')
    all_molecules_filepath = glob.glob('data/rule_of_thumb/n = 75000/2022*')
    # all_molecules_filepath = glob.glob('data/rule_of_thumb/n = 157500/2022*')
    # all_molecules_filepath = glob.glob('data/rule_of_thumb/n = 500000/2022*')
    all_molecules_filepath_small = glob.glob('data/rule_of_thumb/estimate_slope, nmax = 31 500/*')
    for path in all_molecules_filepath:
        dict_molecule = pickle.load(open(path, 'rb'))
        for path_small in all_molecules_filepath_small:         # match molecules from both files
            dict_molecule_small = pickle.load(open(path_small, 'rb'))
            if dict_molecule_small['dataset_name'] == dict_molecule['dataset_name']:
                break
        # plot_data.measure_slope(dict_train=dict_molecule)       # predicts hyperparams for rule of thumb
        plot_data.rule_of_thumb(dict_train=dict_molecule_small, dict_test=dict_molecule)

        # plot_data.computational_cost_molecule(dict_molecule)    # plots runtimes with optimal preconsizes
        # plot_data.measure_optimal_k(dict_train=dict_molecule)   # calculates optimal k for different methods

    plot_data.bar_plot_rule_of_thumb(path_to_data=folder_data + '/rule_of_thumb.csv')
    select_matrix_size = [75000, 158000, 500000]
    for n in select_matrix_size:
        plot_data.bar_plot_rule_of_thumb_single_experiment(n=n, path_to_data=folder_data + '/rule_of_thumb.csv')

    # collect CPU/GPU information
    select_matrix_size = [75000, 157500, 500000]
    list_dict_cgpu = [plot_data.get_cpugpu_info([pickle.load(open(path, 'rb')) for path in
                                                      glob.glob(folder_data + f'data/rule_of_thumb/n = {n}/2022*')])
                      for n in select_matrix_size]
    df_all_runs = pd.concat(list_dict_cgpu)
    # print(df_all_runs.to_latex(columns=['Ethanol', 'Uracil'], sparsify=True))
    # print(df_all_runs.to_latex(columns=['Toluene', 'Aspirin'], sparsify=True))
    # print(df_all_runs.to_latex(columns=['Azobenzene', 'Catcher'], sparsify=True))
    # print(df_all_runs.to_latex(columns=['Nanotube'], sparsify=True))

    # print(df_all_runs.to_latex(columns=['Ethanol', 'Uracil', 'Toluene', 'Aspirin'], sparsify=True))
    # print(df_all_runs.to_latex(columns=['Azobenzene', 'Catcher', 'Nanotube'], sparsify=True))


    # calculate matrix size
    # n_kernel,  k_size = 1000 * 63, int(25_000)
    # size_MG = memory_demand_matrix(int(0.1*n_kernel), n_kernel)
    # print(f'L.shape = ({n_kernel}, {k_size}), memory size: {size_MG:.1f}MB or {size_MG* 2**-10:.1f}GB')

