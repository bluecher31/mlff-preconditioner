import numpy as np

import argparse

from src.tools import create_data
from typing import Dict, List, Union, TypeVar


def initialize_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='iterator over all possible combinations.')
    # parser.add_argument('--repetitions', type=int, default=1, help='Run several repetitions for preconditioners.')

    # general input
    parser.add_argument('--n_measurements', default=1)
    parser.add_argument('--name_dataset', default=['nanotube'], nargs='+',
                        choices=['aspirin', 'uracil', 'ethanol', 'azobenzene', 'toluene',
                                 'catcher', 'nanotube'])
    parser.add_argument('--min_columns', type=int, default=250, help='absolut number of columns used as smallest'
                                                                    'preconditioner')
    parser.add_argument('--max_percentage', default=0.5, type=float, help='relative number of columns in percentage, '
                                                                        'largest preconditioner used')
    parser.add_argument('--absolut_path_to_script', type=str, default='./')
    parser.add_argument('--datapoint_distr', default='linear', choices=['linear', 'log'], type=str)

    # cg_steps
    parser.add_argument('--n_datapoints', default=20, type=int)
    # create_data.cg_steps(n_datapoints=25, n_measurements=5, max_percentage=1, min_columns=120, name_dataset='aspirin',
    #                      flag_eigvals=True, flags=flags)
    #  choices=['eigvec_precon', 'lev_random', 'cholesky', 'random_scores', 'lev_scores', 'inverse_lev',
    #  'truncated_cholesky_custom', 'rank_k_lev_scores']
    parser.add_argument('--preconditioner', type=str, nargs='+',
                        default=['cholesky']
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


# class HyperParamsGenerator:
#     def __init__(self, dict_params: Dict[str, Set]):
#         self.combinations: List[Dict] = []
#         for key in dict_params:
#             options = dict_params[key]
#             self.combinations.append({key: options})
#             raise NotImplementedError()
#
#     def get_hyperparams(self, index: int) -> Dict:
#         assert 0 <= index <= len(self.combinations)
#
#         return self.combinations[index]


def create_list_percentage(task: Dict, datapoint_distr: str, k_minimal: int, max_relative_strength: float,
                           n_measurements: int) -> List[float]:
    assert k_minimal >= 0
    assert 0 < max_relative_strength <= 1
    assert n_measurements > 0
    # determine spacing preconditioner strength
    n_columns = task['F_train'].size        # number of columns in K, i.e. K.shape = (n_columns, n_columns)
    # n_measurements: int, max_percentage: float, min_columns: float
    min_percentage = k_minimal / n_columns
    assert min_percentage < max_relative_strength
    if datapoint_distr == 'linear':
        list_percentage = np.linspace(max_relative_strength, min_percentage, n_measurements)  # begin with strongest precond.
    elif datapoint_distr == 'log':
        list_percentage = np.geomspace(max_relative_strength, min_percentage, n_measurements)  # begin with strongest precond.
    else:
        assert False, f'incorrect keyword: {datapoint_distr}'
    return list(list_percentage)


Value = TypeVar('Value', float, int)


def select_value(values: Union[Value, List[Value]], index: int, n_options: int) -> [Value, int]:
    """
    Selects a unique value and from List and returns an updated index (for later use on  additional lists).
    """
    if isinstance(values, int) or isinstance(values, float):
        return values, index, n_options
    else:
        i_rest, i_current = divmod(index, len(values))       # quotient, remainder
        value = values[i_current]
        n_options *= len(values)
        return value, i_rest, n_options


if __name__ == '__main__':
    args = initialize_parser()
    # rsync -avz mlc:~/Projects/cholesky/project/data_new/ /Users/sbluecher/Documents/University/git/cholesky/project/data_new/ --include='*.pickle'

    # NEW
    # unique_experiment_name/preconditioner/k=xxx
    # store runtimes by default -> to check whether this is compatible with previous set-up
    #                               maybe FIX GPU globally to render everything compatible


    index = args.index
    for index in range(1):
        preconditioner_str, i_interal, n_options = select_value(values=args.preconditioner, index=index, n_options=1)
        name_dataset, i_interal, n_options = select_value(values=args.name_dataset, index=i_interal, n_options=n_options)

        n_datapoints, i_interal, n_options = select_value(values=args.n_datapoints, index=i_interal,
                                                          n_options=n_options)

        # use size equivalent to aspirin
        n_datapoints = create_data.normalize_to_aspirin(n_datapoints=n_datapoints, dataset_name=name_dataset)
        print(f'dataset = {name_dataset} \nn_datapoints = {n_datapoints}')

        task, gdml_train = create_data.create_task(n_datapoints=n_datapoints, path_to_script=args.absolut_path_to_script,
                                                   name_dataset=name_dataset)

        np_percentage = create_list_percentage(task=task,
                                               datapoint_distr=args.datapoint_distr,
                                               n_measurements=int(args.n_measurements),
                                               max_relative_strength=float(args.max_percentage),
                                               k_minimal=args.min_columns)

        relative_preconditioner_strength, _, n_options = select_value(values=list(np_percentage), index=i_interal,
                                                                      n_options=n_options)

        print(f'Running task {index} from a total of {n_options}.')
        assert n_options > index, 'All possible options have been exhausted.'
        print('eigvals', args.calculate_eigvals)
        create_data.cg_steps(task=task, gdml_train=gdml_train,
                             n_datapoints=n_datapoints,
                             flag_eigvals=args.calculate_eigvals,
                             path_to_script=args.absolut_path_to_script, preconditioner=preconditioner_str,
                             preconditioner_strength=float(relative_preconditioner_strength))
        del gdml_train

    # test in terminal
    # qlogin -binding linear:4 -l cuda=1

    # not sure if there is a direct way but you can explicitly specify the types of GPU that you want to use:
    # -l gputype='P100*|V100|A100*'
    #
    # or if you want to exclude certain GPUs with small VRAM:
    # -l gputype='!(P100G12|GTX1080)'
    #
    # here, | is the or-operator and ! negates the statement
    # and A100* encompasses A100G40 and A100G80 etc.

    # sgdml test model.npz ethanol_ccsd_t-test.npz 100
    # sgdml test model.npz aspirin_dft.npz 100
