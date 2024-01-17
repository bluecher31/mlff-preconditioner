import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd

import pickle
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from functools import partial
from .cluster_information import convert_node_to_gpu, convert_node_to_cpu

from typing import List, Dict, Union

# color_vertical_cost_line = '#CC78BC'        # old color, equals C5
# color_vertical_cost_line = f'C6'
color_vertical_cost_line = 'grey'


def map_dict_label_to_paper(label_preconditioner: str):
    switch = {
        'eigvec_precon': 'Optimal SVD',
        'lev_random': 'Leverage scores',
        'cholesky': 'incomplete Cholesky',
        'random_scores': 'Uniform sampling',
        'lev_scores': r'Deterministic $\bar{\tau}$',
        'inverse_lev': r'Reversely ordered $\bar{\tau}$',
        'eigvec_precon_block_diagonal': 'Naive Jacobi',
        'eigvec_precon_atomic_interactions': 'Advanced Jacobi',
        'truncated_cholesky': 'Truncated Cholesky',
        'truncated_cholesky_custom': 'Truncated Cholesky (custom)',
        'rank_k_lev_scores': r'$k$-leverage scores'
    }
    return switch[label_preconditioner]


def map_dict_label_to_color(label_preconditioner: str):
    switch = {
        'eigvec_precon': f'C0',
        'lev_random': f'C1',
        'cholesky': f'C2',
        'random_scores': f'C3',
        'lev_scores': f'C8',
        'inverse_lev': f'C9',
        'eigvec_precon_block_diagonal': 'C4',
        'eigvec_precon_atomic_interactions': 'C6',
        'truncated_cholesky': 'brown',
        'truncated_cholesky_custom': 'navy',
        'rank_k_lev_scores': 'C5'
    }
    return switch[label_preconditioner]


def map_dataset_name_to_molecule(dataset_name: str, reverse_order: bool = False):
    """Maps molecule to dataset_name (toluene -> C6H5CH3). If reverse_order=True opposite (C6H5CH3 -> toluene)."""
    # list_methods_plot = ['aims_catcher', 'larger_aims_nanotube']
    switch = {
        'aims_catcher': f'Catcher',
        'larger_aims_nanotube': f'Nanotube',
        'azobenzene_new': 'Azobenzene',
        'C6H5CH3': 'Toluene',
    }
    if reverse_order is True:
        switch = {val.lower(): key for (key, val) in switch.items()}
    if dataset_name in switch:
        molecule_name = switch[dataset_name]
    else:
        molecule_name = dataset_name.capitalize()
    return molecule_name


def map_z_to_atom_type(z: np.ndarray):
    switch = {
        1: 'H',
        6: 'C',
        8: 'O',
    }
    list_atom_type = [switch[atom] for atom in z]
    return np.array(list_atom_type)


def infer_optimal_k_RoT(n_kernel: int, const_CG: float, slope: float):
    return (const_CG * slope * n_kernel**2 / 2)**(1/(2 + slope))


def calculate_error(x: np.ndarray, y: np.ndarray, return_error: bool) -> [np.ndarray, np.ndarray, bool, int]:
    """Detect identical measurements and compute there average. """
    x_unique = np.unique(x)
    n_min = 1
    if np.alltrue(x_unique == x):
        return x, y, False, n_min
    else:
        list_values = [y[x == x_tmp] for x_tmp in x_unique]
        if return_error is False:
            y_new = np.array([np.mean(values) for values in list_values])
        else:
            y_new = np.array([np.std(values)/np.sqrt(len(values)) for values in list_values])
            # y_new = np.std(values, axis=1) / np.sqrt(values.shape[1])
        # means = [np.mean(y[x == x_tmp]) for x_tmp in x_unique]
        assert x_unique.shape == y_new.shape
        n_measurments = [len(values) for values in list_values]
        # y_mean = np.stack(means)
        return x_unique, y_new, True, min(n_measurments)


def plot_plane_cg_steps(ax: plt.Axes, percentage: np.ndarray, cg_steps: np.ndarray, max_percentage: float,
                        n_kernel: int, label: str):
    """Line plot of cg steps, additionally errors are shown as a band with width=std."""
    assert 0 < max_percentage <= 1
    mask = np.bitwise_and(percentage < max_percentage, cg_steps < n_kernel * 5)
    x = percentage[mask]
    y = cg_steps[mask]

    x_mean, y_mean, error_bars, _ = calculate_error(x, y, return_error=False)
    # plt.plot(x * 100, y, '-x', c=map_dict_label_to_color(label), label=map_dict_label_to_paper(label), markersize=3)

    [line_plot] = ax.plot(x_mean * 100, y_mean, '-x', c=map_dict_label_to_color(label),
                          label=map_dict_label_to_paper(label), markersize=3)
    if error_bars is True:
        _, y_err, _, _ = calculate_error(x, y, return_error=True)
        # plt.errorbar(x=x_mean * 100, y=y_mean, yerr=y_err, fmt='-x', c=map_dict_label_to_color(label),
        #              label=map_dict_label_to_paper(label), markersize=3)

        ax.fill_between(x_mean * 100, y_mean - y_err, y_mean + y_err,
                        color=line_plot.get_color(), alpha=0.2)


def cg_steps(dict_data: dict, title: str = '', plot_save=False, max_percentage: float = 1.):
    if title == 'det_lev_scores':
        all_labels = ['lev_random', 'lev_scores', 'inverse_lev']
        title = title + '_'
        figsize = (4, 2.2)
    else:
        all_labels = ['eigvec_precon', 'lev_random', 'cholesky', 'random_scores', 'lev_scores', 'inverse_lev',
                      'truncated_cholesky', 'truncated_cholesky_custom', 'rank_k_lev_scores']
        figsize = (8, 4)
    # for label in labels:
    labels = [label for label in all_labels if f'{label}_percentage' and f'{label}_cgsteps' in dict_data]

    n_kernel = dict_data['K.shape'][0]       # size kernel matrix
    dataset = str(dict_data["dataset_name"])

    fig = plt.figure(f'{title}{dataset} - n = {n_kernel}', figsize=figsize)
    # plt.xlim([0, 15])
    ax = fig.subplots()

    for label in labels:
        plot_plane_cg_steps(ax=ax, percentage=dict_data[f'{label}_percentage'], cg_steps=dict_data[f'{label}_cgsteps'],
                            max_percentage=max_percentage, n_kernel=n_kernel, label=label)

    # plt.title(f'{dataset} K.shape = {dict_data["K.shape"]}')
    # plt.legend()

    plt.legend(ncol=1, loc='center right')

    plt.xlabel(r'percentage of columns $\frac{k}{n}$')
    plt.ylabel(r'# iteration')
    plt.semilogy()
    # plt.grid()


    ax = plt.gca()

    # ax.set_title(f'{dataset} K.shape = {dict_data["K.shape"]}')
    text_str = f'{dataset.capitalize()}, n = {n_kernel}'
    at = AnchoredText(text_str, prop=dict(size=10), frameon=True, loc='upper right', pad=0.5)
    at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
    ax.add_artist(at)

    ax.axhline(y=n_kernel, linewidth=1, color=color_vertical_cost_line)
    # ax.axhline(y=0.25 * n_kernel, linewidth=1, color=c)
    ax.axhline(y=0.1 * n_kernel, linewidth=1, color=color_vertical_cost_line, linestyle='dashed')


    # # add second axis with absolute number of columns k
    # ax1 = plt.gca()
    # ax2 = ax1.twiny()
    # ax2.set_xticks(ax1.get_xticks())
    # ax2.set_xbound(ax1.get_xbound())
    # ax2.set_xticklabels([f'{x/100 * n_kernel:.0f}' for x in ax1.get_xticks()])

    plt.tight_layout(pad=0.1)
    if plot_save is True:
        file_name = f'cg_steps_{dataset}_{dict_data["K.shape"][0]}'
        plt.savefig(file_name)


def time_cholesky(dic_data: dict):
    time_cholesky = dic_data['t_cholesky']

    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved_10 = np.convolve(time_cholesky, kernel, mode='same')

    plt.figure('time_cholesky')
    plt.title(f'K.shape = {dic_data["K.shape"]}, k = {len(time_cholesky)}')
    plt.plot(time_cholesky, label='original')
    plt.plot(data_convolved_10, label=f'conv{kernel_size}')
    plt.ylabel('time in s')
    plt.xlabel('number of cholesky updates')
    plt.axhline(dic_data['time_cg_step'], xmin=0, label='cg', c='red')

    plt.legend()
    plt.tight_layout(pad=0.1)


def preprocess_eigvals(raw_eigvals, n_eigvals: int = None):
    normalized_eigvals = np.abs(raw_eigvals) / np.abs(raw_eigvals).min()
    normalized_eigvals = np.sort(normalized_eigvals)[::-1]
    return normalized_eigvals[:n_eigvals]


def eigvals_schematic(dict_data: dict):
    """Only plot original eigvals for schematic."""
    label = 'random_scores'
    raw_eigvals = dict_data[f'eigvals_{label}_0']
    normalized_eigvals = preprocess_eigvals(raw_eigvals)
    plt.figure('plane_raw_eigvals')
    plt.plot(normalized_eigvals, color='#CC0000')
    ax = plt.gca()
    plt.axis(False)
    # ax.tick_params(left=True,
    #                bottom=True,
    #                labelleft=False,
    #                labelbottom=False)
    plt.semilogy()
    plt.tight_layout(pad=0.1)



def eigvals(dict_data: dict, n_eigvals: int = 150,
            all_labels=['eigvec_precon', 'lev_random', 'cholesky', 'random_scores', 'lev_scores', 'inverse_lev', 'truncated_cholesky']):

    # all_labels = ['eigvec_precon', 'random_scores', 'lev_random', 'cholesky',
    #               'eigvec_precon_block_diagonal', 'eigvec_precon_atomic_interactions']

    labels = [label for label in all_labels if f'{label}_percentage' and f'{label}_cgsteps' in dict_data]
    print(f'Found {len(labels)} different preconditioners')
    dataset = str(dict_data["dataset_name"])
    n_kernel = dict_data['K.shape'][0]  # size kernel matrix
    text_str = f'{dataset.capitalize()}, n = {n_kernel}'

    # extract percentage
    percentage = []
    for key in dict_data.keys():
        target_preconditioner = labels[0]
        str_eigvals = f'eigvals_{target_preconditioner}_'
        if str_eigvals == key[:len(str_eigvals)]:
            key_splitted = key.split('_')
            current_percentage = float(key_splitted[-1])
            if current_percentage > 0.:  # do not include raw preconditioner
                percentage.append(current_percentage)

        # if key not in labels:
        #     key_splitted = key.split('_')
        #     test_label_splitted = labels[0].split('_')
        #
        #     if key_splitted[0] == 'eigvals':
        #         print('test')
        #         if test_label_splitted[0] == key_splitted[1] \
        #                 and test_label_splitted[1] == key_splitted[2] \
        #                 and len(test_label_splitted) + 2 == len(key_splitted):
        #             current_percentage = float(key_splitted[-1])
        #             if current_percentage > 0.:            # do not include raw preconditioner
        #                     percentage.append(current_percentage)

    # remove some measurements manually
    # percentage = [percentage[0], percentage[-2], percentage[-1]]
    # percentage = percentage[::4]
    # percentage = percentage[-4:]

    # original configuration below
    if dataset == 'larger_aims_nanotube':
        percentage = percentage[::2]      # nanotube
    # elif 'eigvec_precon_block_diagonal' in labels:
    #     percentage = percentage[:1]
    #     n_eigvals = n_kernel
    #     labels.remove('eigvec_precon')
    else:
        percentage = percentage[::4]

    n = len(percentage)
    if n == 0:
        assert False, 'no measurements found'
    print(f'Found n = {n} measurements')

    figsize = (n * 1.7 + 1.5, 1.95)
    # fig = plt.figure('eigvals_'+text_str, figsize=(6.88, 3.5/2.))
    # fig = plt.figure('eigvals_' + text_str, figsize=(n*1.7 + 3.5, 2.9))
    fig = plt.figure('eigvals_' + text_str, figsize=figsize)
    # fig = plt.figure('eigvals_' + text_str, figsize=(8.5, 1.5))
    for i, p in enumerate(percentage[::-1]):
        if i == 0:
            ax = plt.subplot(1, n, i+1)
            ax0 = ax
        else:
            ax = plt.subplot(1, n, i + 1, sharex=ax0, sharey=ax0)

        # ax.set_title(f'preconditioning: {p:.2f} % ')    #, fontdict={'fontsize': formatter.fontsizes.small})



        ax.tick_params(left=False,
                       bottom=True,
                       labelleft=False,
                       labelbottom=True)
        raw_eigvals = dict_data[f'eigvals_{labels[0]}_0']

        # normalized_eigvals = np.abs(raw_eigvals)[:n_eigvals]/np.abs(raw_eigvals).min()
        # normalized_eigvals = np.sort(normalized_eigvals)[::-1]      # sort from large to small
        ax.plot(preprocess_eigvals(raw_eigvals, n_eigvals), c='grey', alpha=0.5)

        for j_label, label in enumerate(labels):
            key = f'eigvals_{label}_{p:.2f}'
            eigvals = dict_data[key]
            if j_label == 0:
                k_array = len(eigvals) * dict_data[f'{label}_percentage']
                k_print = k_array[::-1][i]
                k_print_calculate = int(p/100. * n_kernel)
                k_str = f'k = {k_print_calculate:.0f}'
                at = AnchoredText(
                    k_str, prop=dict(size=10), frameon=True, loc='upper right', pad=0.5)
                at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
                if n_eigvals != n_kernel:
                    ax.set_title(k_str)
                # ax.add_artist(at)
            else:
                k_array = len(eigvals) * dict_data[f'{label}_percentage']
                k = k_array[::-1][i]
                assert k_print == k, 'only exact exactly equal preconditioners in terms of k'

            # str_label = map_dict_label_to_paper(label) + f'  {k:.1f}'             # full details
            str_label = map_dict_label_to_paper(label)
            ax.plot(preprocess_eigvals(eigvals, n_eigvals), c=map_dict_label_to_color(label), label=str_label)
            # plt.legend()
        if i == len(percentage)-1:
            plt.legend(ncol=1, loc='upper right')
            # plt.legend(ncol=4, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            #                   mode="expand", borderaxespad=0.)          # $n_\mathrm{train}$',    , title=r'$n$'
            pass

        if i == int((n)/2):
            ax = plt.gca()

            # ax.set_title(f'{dataset} K.shape = {dict_data["K.shape"]}')
            at = AnchoredText(text_str, prop=dict(size=10), frameon=True, loc='upper left', pad=0.5)
            at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
            # ax.add_artist(at)
            # ax.set_title(text_str, size=10)
            ax.set_xlabel(r'# eigenvalues')
        if i == 0:
            # ax.set_ylabel(r'Condition number $\kappa$')
            ax.set_ylabel(r'Spectrum of $P^{-1}K_{\lambda}$')
            ax.tick_params(left=True,
                           bottom=True,
                           labelleft=True,
                           labelbottom=True)
            kappa = float(np.abs(np.max(preprocess_eigvals(raw_eigvals, n_eigvals))))
            text_str = 'Condition number\n $\kappa=' + f'{kappa:.1E}' + '$'
            at = AnchoredText(text_str, prop=dict(size=8, color='grey'), frameon=False, loc='lower left', pad=0.5)
            at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
            ax.add_artist(at)

        ax.semilogy()

    # plt.title('')
    # fig.legend(ncol=4,bbox_to_anchor=(0., 0.92, 1., .102), loc='upper center')
    plt.tight_layout(pad=0.1)           # change overall padding to be tight
    plt.subplots_adjust(wspace=0)       # change padding between all subplots, call after tight_layout()
    # plt.savefig('eigvals.pdf')


def cg_performance_multiple_molecules(list_data_dict: List[dict], selection: str, max_percentage: float = 1.):
    if selection == 'paper_old':
        all_labels = ['eigvec_precon', 'random_scores', 'lev_random', 'cholesky']
        list_methods_plot = ['ethanol', 'aspirin', 'aims_catcher', 'larger_aims_nanotube']
        list_data_dict = [d for d in list_data_dict
                          if np.sum([d['dataset_name'] == name for name in list_methods_plot]) == 1]
    elif selection == 'all_old':
        all_labels = ['eigvec_precon', 'random_scores', 'lev_random', 'cholesky', 'lev_scores', 'inverse_lev']
    elif selection == 'new':
        all_labels = ['eigvec_precon', 'random_scores', 'lev_random', 'cholesky', 'lev_scores', 'inverse_lev',
                      'truncated_cholesky', 'rank_k_lev_scores']
    else:
        raise ValueError(f'selection = {selection} not defined. ')

    figsize = (8, 1.2)

    list_data_dict.sort(key=lambda d: d['n_datapoints'], reverse=True)
    n_methods = len(list_data_dict)
    figsize = (figsize[0], n_methods * figsize[1])

    fig = plt.figure(f'cg_performance_{list_data_dict[0]["K.shape"][0]}', figsize=figsize)

    for index, dict_data in enumerate(list_data_dict):
        n_kernel = dict_data['K.shape'][0]  # size kernel matrix
        dataset = str(dict_data["dataset_name"])

        labels = [label for label in all_labels if f'{label}_percentage' and f'{label}_cgsteps' in dict_data]

        if index == 0:
            ax = plt.subplot(n_methods, 1, index+1)
            ax0 = ax
        else:
            ax = plt.subplot(n_methods, 1, index + 1, sharex=ax0)

        # plot performance for all methods
        for label in labels:
            plot_plane_cg_steps(ax=ax, percentage=dict_data[f'{label}_percentage'],
                                cg_steps=dict_data[f'{label}_cgsteps'],
                                max_percentage=max_percentage, n_kernel=n_kernel, label=label)

        # ax.set_title(f'{dataset} K.shape = {dict_data["K.shape"]}')
        # text_str = f'{dataset.capitalize()}, n = {n_kernel}'
        print(dataset)
        text_str = f'{map_dataset_name_to_molecule(dataset)}'
        at = AnchoredText(text_str, prop=dict(size=10), frameon=True, loc='upper right', pad=0.5)
        at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
        ax.add_artist(at)

        if ax.get_ylim()[1] > n_kernel * 0.9:       # plot n**3 only if cg run was as costly
            ax.axhline(y=n_kernel, linewidth=1, color=color_vertical_cost_line)
        # ax.axhline(y=0.25 * n_kernel, linewidth=1, color=c)
        ax.axhline(y=0.1 * n_kernel, linewidth=1, color=color_vertical_cost_line, linestyle='dashed')



        ax.semilogy()
        x_pos_text = 0.9
        if index == n_methods-1:
            # plt.legend(ncol=2, loc=(0.585, 0.285))
            plt.legend(ncol=2, loc='lower left')
            ax.set_xlabel(r'relative preconditioner strength $\frac{k}{n}$')
        elif index == int(n_methods/2):
            ax.set_ylabel('$\#$ iterations')
            len_xaxis = ax.get_xlim()[1] - ax.get_xlim()[0]
            # plt.text(len_xaxis* x_pos_text, n_kernel * 0.11, '10% reduction      ', rotation=360, color=color_vertical_cost_line,
            #          horizontalalignment='right')
            # ax.set_ylim(ax.get_ylim()[0], 1.5 * ax.get_ylim()[1])
        elif dataset == 'aspirin':
            len_xaxis = ax.get_xlim()[1] - ax.get_xlim()[0]
            # plt.text(len_xaxis / 2 * 0.5, n_kernel * 1.1, f'full cholesky decomposition: $n^3$', rotation=360, color=color_vertical_cost_line)
            plt.text(len_xaxis * x_pos_text * 0.95, n_kernel * 1.1, r'Closed-form solve: $n^3$', rotation=360, ha="right",
                     color=color_vertical_cost_line)
            plt.text(len_xaxis * 0.23, n_kernel * 0.11, '10% reduction      ', rotation=360,
                     color=color_vertical_cost_line,
                     horizontalalignment='right')
        else:
            ax.tick_params(left=True,
                           bottom=False,
                           labelleft=True,
                           labelbottom=False)

    plt.tight_layout(pad=0.1)           # change overall padding to be tight
    plt.subplots_adjust(hspace=0)       # change padding between all subplots, call after tight_layout()


    # ax_big = fig.add_subplot(111, frameon=False)
    # # hide tick and tick label of the big axis
    # ax_big.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # ax_big.set_xlabel('% of all columns/memory')
    # ax_big.set_ylabel('cg steps/computational')
    # # fig.text(0.04, 0.5, 'cg steps/computational', va='center', rotation='vertical')
    # fig.tight_layout(pad=0.1)


def preconditioner_size(dict_data: dict):
    """Visualize linear dependence between preconditioner size k and normalized cg iterations. for a single molecule (Aspirin)"""
    all_labels = ['eigvec_precon', 'lev_random', 'cholesky', 'random_scores', 'lev_scores', 'inverse_lev']

    list_n_datapoints = dict_data['list_n_datapoints']


    labels = [label for label in all_labels if f'{list_n_datapoints[0]}_{label}_percentage' and f'{list_n_datapoints[0]}_{label}_cgsteps' in dict_data]
    assert len(labels) == 1, 'only consistent preconditioner within this experiment'

    label_preconditioner = labels[0]

    figsize = (4., 3.5)
    fig = plt.figure('preconditioner_size', figsize=figsize)
    for n_datapoints in list_n_datapoints:
        n_kernel = dict_data[f'{n_datapoints}_K.shape'][0]  # size kernel matrix
        x = dict_data[f'{n_datapoints}_{label_preconditioner}_percentage']
        y = dict_data[f'{n_datapoints}_{label_preconditioner}_cgsteps']

        # plt.plot(x * 100, y / n_kernel, '-x', label=f'{n_datapoints*63}')
        plt.plot(x*n_kernel, y / n_kernel, '--.', label=f'{n_kernel}', linewidth=1)


    dataset = dict_data["dataset_name"]

    # plt.title(f'{dataset} K.shape = {dict_data["K.shape"]}')
    # plt.legend()


    # plt.xlabel(r'percentage of columns $\frac{k}{n}$')
    plt.xlabel(r'inducing columns ${k}$')
    plt.ylabel(r'normalized iteration $\frac{\#}{n}$')
    # plt.semilogy()
    plt.loglog()
    # plt.grid()

    dataset = str(dict_data["dataset_name"])
    ax = plt.gca()

    # ax.set_title(f'{dataset} K.shape = {dict_data["K.shape"]}')
    text_str = f'{dataset.capitalize()}'
    at = AnchoredText(text_str, prop=dict(size=10), frameon=True, loc='upper right', pad=0.5)
    at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
    ax.add_artist(at)

    c = '#CC78BC'
    ax.axhline(y=1, linewidth=1, color=color_vertical_cost_line)
    # ax.axhline(y=0.25 * n_kernel, linewidth=1, color=c)
    ax.axhline(y=0.1 * 1, linewidth=1, color=color_vertical_cost_line, linestyle='dashed')

    # corrected upper performance bound n - k
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_numerical_error = np.linspace(xmin-50, xmax+1000, 100)
    ax.fill_between(x_numerical_error, 1, 150, hatch="//", edgecolor=color_vertical_cost_line, facecolor='white')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    k_rule = np.linspace(xmin, xmax, 100)
    slope = 1.25

    def linear_rule_of_thumb(k, prefactor: float, slope: float,):
        return prefactor * k ** (-slope)

    params_slope, pcov = curve_fit(f=partial(linear_rule_of_thumb, slope=slope),
                                   xdata=x * n_kernel, ydata=y / n_kernel,
                                   sigma=y / n_kernel * 0.05,
                                   # p0=[1],)
                                   bounds=(0.0001, np.inf))  # bounds=([-10, 0, 0], [0, 1500, 1]),
    # plt.figure()
    # plt.plot(k_rule,
    #          linear_rule_of_thumb(k_rule, slope=slope, prefactor=params_slope[0]),
    #          label=f'fit: {n_kernel}')
    # plt.loglog()

    # plt.legend(ncol=1, loc=(1.02, 0.1), title='kernel size $n$', title_fontsize=9)
    plt.legend(ncol=3, loc='lower left', title='kernel size $n$', title_fontsize=9)          # $n_\mathrm{train}$',    , title=r'$n$'

    # plt.legend(ncol=3, loc='lower left')          # $n_\mathrm{train}$',    , title=r'$n$'
    # plt.legend(ncol=3, loc='center right')  # $n_\mathrm{train}$',    , title=r'$n$'
    plt.tight_layout(pad=0.1)


    # # add second axis with absolute number of columns k
    # ax1 = plt.gca()
    # ax2 = ax1.twiny()
    # ax2.set_xticks(ax1.get_xticks())
    # ax2.set_xbound(ax1.get_xbound())
    # ax2.set_xticklabels([f'{x/100 * n_kernel:.0f}' for x in ax1.get_xticks()])


def preconditioner_size_molecules(list_dict_molecules: List[dict]):
    """Linear dependence between preconditioner size k and normalized cg iterations for a different molecules."""
    # list_n_datapoints = dict_data['list_n_datapoints']
    figsize = (4., 3.5)
    fig = plt.figure('preconsize_different_molecules', figsize=figsize)
    label_preconditioner = None
    for dict_molecule in list_dict_molecules:
        all_labels = ['eigvec_precon', 'lev_random', 'cholesky', 'random_scores', 'lev_scores', 'inverse_lev']
        n_datapoints = max(dict_molecule['list_n_datapoints'])
        labels = [label for label in all_labels if f'{n_datapoints}_{label}_percentage' and f'{n_datapoints}_{label}_cgsteps' in dict_molecule]
        assert len(labels) == 1, 'only consistent preconditioner within this experiment'
        if label_preconditioner is None or label_preconditioner == labels[0]:
            label_preconditioner = labels[0]
        else:
            assert False, 'Expect constistent preconditioner between all dict_molecules.'

        n_kernel = dict_molecule[f'{n_datapoints}_K.shape'][0]  # size kernel matrix
        dataset_name = str(dict_molecule["dataset_name"])

        x = dict_molecule[f'{n_datapoints}_{label_preconditioner}_percentage']
        y = dict_molecule[f'{n_datapoints}_{label_preconditioner}_cgsteps']

        mask = x < 0.7
        # plt.plot(x * 100, y / n_kernel, '-x', label=f'{n_datapoints*63}')
        plt.plot(x[mask] * n_kernel, y[mask] / n_kernel, '--.', label=f'{map_dataset_name_to_molecule(dataset_name)}',
                 linewidth=1)

    # plt.xlabel(r'percentage of columns $\frac{k}{n}$')
    plt.xlabel(r'inducing columns ${k}$')
    plt.ylabel(r'normalized iteration $\frac{\#}{n}$')
    # plt.semilogy()
    plt.loglog()
    # plt.grid()

    ax = plt.gca()
    #
    # # ax.set_title(f'{dataset} K.shape = {dict_data["K.shape"]}')
    # text_str = f'{dataset.capitalize()}'
    # at = AnchoredText(text_str, prop=dict(size=10), frameon=True, loc='upper right', pad=0.5)
    # at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
    # ax.add_artist(at)

    c = '#CC78BC'
    ax.axhline(y=1, linewidth=1, color=color_vertical_cost_line)
    # ax.axhline(y=0.25 * n_kernel, linewidth=1, color=c)
    ax.axhline(y=0.1 * 1, linewidth=1, color=color_vertical_cost_line, linestyle='dashed')

    # corrected upper performance bound n - k
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_numerical_error = np.linspace(xmin-50, xmax+1000, 100)
    ax.fill_between(x_numerical_error, 1, 150, hatch="//", edgecolor=color_vertical_cost_line, facecolor='white')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # plt.legend(ncol=1, loc=(1.02, 0.1), title=f'n = {np.round(n_kernel, -2)}', title_fontsize=9)
    plt.legend(ncol=2, loc='upper right', title=f'$n$ = {np.round(n_kernel, -2)}', title_fontsize=9)
    plt.tight_layout(pad=0.1)


def calculate_jackknife(measurements: np.ndarray):
    mask = np.zeros(measurements.shape, dtype=np.bool)
    jackknife_mean_list = []
    for i in range(len(measurements)):
        mask[i] = True
        assert mask.sum() == 1, 'only remove a single measurement'
        jackknife_mean_list.append(measurements[mask==False].mean())
        mask[i] = False
    mean_estimates = np.array(jackknife_mean_list)
    return mean_estimates.mean(), mean_estimates.std()


def rule_of_thumb_fn(k_column: np.ndarray, slope: float, prefactor: float, k_unity: float,
                     n_kernel_rule: int):
    return prefactor * (k_column / k_unity) ** (-slope) + (k_column / n_kernel_rule) ** 2


def measure_slope(dict_train: Dict):
    name_dataset = dict_train["dataset_name"]
    plt.figure(f'slope - {name_dataset}')
    list_slope = []
    list_k_unity = []
    # for n_datapoints_fit_rule in dict_train['list_n_datapoints']:     # use all measurements, old

    for n_datapoints_fit_rule in [max(dict_train['list_n_datapoints']) for _ in range(2)]:  # only use largest kernel size n
        n_kernel = dict_train[f'{n_datapoints_fit_rule}_K.shape'][0]
        k_fit_rule = dict_train[f'{n_datapoints_fit_rule}_lev_random_percentage'] * n_kernel
        cg_steps_normalized = dict_train[f'{n_datapoints_fit_rule}_lev_random_cgsteps'] / n_kernel
        time_solve = dict_train[f'{n_datapoints_fit_rule}_lev_random_total_time_solve']

        def fn_linear_cg_slope(k_column: np.ndarray, slope: float, k_unity: float,
                               n_kernel_rule: int):
            # slope = 1
            # prefactor = 1.e3
            # k_unity: naive parity between CG and closed form solve
            return (k_column / k_unity) ** (-slope)
        # a = partial(rule_of_thumb, n_kernel_rule=n_kernel)
        mask = k_fit_rule/n_kernel < 0.7
        params_slope, pcov = curve_fit(f=partial(fn_linear_cg_slope, n_kernel_rule=n_kernel),
                                               xdata=k_fit_rule[mask], ydata=cg_steps_normalized[mask],
                                       sigma=cg_steps_normalized[mask]*0.05,
                                               # p0=[1, 1, 1],
                                               bounds=(0.0001, np.inf))  # bounds=([-10, 0, 0], [0, 1500, 1]),
        list_slope.append(params_slope[0])
        list_k_unity.append(params_slope[1])
        # print(f'n_datapoints = {n_datapoints_fit_rule}\n params = {params_slope}')
        plt.plot(k_fit_rule, cg_steps_normalized, 'x', label=n_datapoints_fit_rule)
        plt.plot(k_fit_rule, fn_linear_cg_slope(k_fit_rule, slope=params_slope[0], k_unity=params_slope[1],
                                                n_kernel_rule=n_kernel), label=f'fit: {n_datapoints_fit_rule}')
    plt.loglog()
    plt.legend()

    # calculate mean and variance via jackknife
    mean_slope, error_slope = calculate_jackknife(measurements=np.array(list_slope))
    mean_k_unity, error_k_unity = calculate_jackknife(measurements=np.array(list_k_unity))
    print(f'{name_dataset}\n'
          f'slope = {mean_slope:.2f} +- {error_slope}\n'
          f'k_unity = {mean_k_unity:.0f} +- {error_k_unity}\n')


def get_params(dataset_name: str, old=False) -> [float, int, float]:
    """This function returns the desired hyperparameters for the rule of thumb."""
    prefactor = 1
    if old is False:
        if dataset_name == 'default':
            slope = 1
            k_unity = 100
        elif dataset_name == 'ethanol':
            slope = 0.87
            k_unity = 10
        elif dataset_name == 'uracil':
            slope = 1.07
            k_unity = 32
        elif dataset_name == 'C6H5CH3' or dataset_name == 'toluene':
            slope = 1.01
            k_unity = 44
        elif dataset_name == 'aspirin':
            slope = 1.14
            k_unity = 236
        elif dataset_name == 'azobenzene_new' or dataset_name == 'azobenzene':
            slope = 1.02
            k_unity = 62
        elif dataset_name == 'aims_catcher' or dataset_name == 'catcher':
            slope = 1.02
            k_unity = 316
        elif dataset_name == 'larger_aims_nanotube' or dataset_name == 'nanotube':
            slope = 0.73
            k_unity = 89
        else:
            raise NotImplementedError(f'dataset_name = {dataset_name} is not specified. ')
    elif old is True:
        if dataset_name == 'default':
            slope = 1
            k_unity = 100
        elif dataset_name == 'ethanol':
            slope = 0.93
            k_unity = 19
        elif dataset_name == 'uracil':
            slope = 1.22
            k_unity = 60
        elif dataset_name == 'C6H5CH3' or dataset_name == 'toluene':
            slope = 1.13
            k_unity = 80
        elif dataset_name == 'aspirin':
            slope = 1.25
            k_unity = 290
        elif dataset_name == 'azobenzene_new' or dataset_name == 'azobenzene':
            slope = 1.13
            k_unity = 100
        elif dataset_name == 'aims_catcher' or dataset_name == 'catcher':
            slope = 1.08
            k_unity = 500
        elif dataset_name == 'larger_aims_nanotube' or dataset_name == 'nanotube':
            slope = 0.63
            k_unity = 70
        else:
            raise NotImplementedError(f'dataset_name = {dataset_name} is not specified. ')
    return slope, k_unity, prefactor


def calculate_optimal_precon_k(dict_molecule, n_datapoints: int) -> Dict:
    n_kernel = dict_molecule[f'{n_datapoints}_K.shape'][0]
    k_fit_rule = dict_molecule[f'{n_datapoints}_lev_random_percentage'] * n_kernel
    cg_steps_normalized = dict_molecule[f'{n_datapoints}_lev_random_cgsteps'] / n_kernel
    time_solve = dict_molecule[f'{n_datapoints}_lev_random_total_time_solve']
    time_preconditioner = dict_molecule[f'{n_datapoints}_lev_random_total_time_preconditioner']
    time_cg = dict_molecule[f'{n_datapoints}_lev_random_total_time_cg']

    k_measured_interpolate = np.linspace(k_fit_rule.min() * 1.01, k_fit_rule.max() * 0.999, 10000, dtype=np.int)

    interp_measured_time_fn = interp1d(x=k_fit_rule, y=time_solve)
    interp_measured_precon_fn = interp1d(x=k_fit_rule, y=time_preconditioner)
    interp_measured_cg_fn = interp1d(x=k_fit_rule, y=time_cg)

    interp_time_solve = interp_measured_time_fn(k_measured_interpolate)
    interp_time_precon = interp_measured_precon_fn(k_measured_interpolate)
    interp_time_cg = interp_measured_cg_fn(k_measured_interpolate)

    # measurement
    optimal_measured_k = k_fit_rule[time_solve.argmin()]
    minimal_time_solve = time_solve.min()
    mask_nearly_optimal_k = interp_time_solve < 1.25 * interp_time_solve.min()
    upper_bound_measured_k = k_measured_interpolate[mask_nearly_optimal_k].max()
    lower_bound_measured_k = k_measured_interpolate[mask_nearly_optimal_k].min()

    dict_optimal_k = {'k_measurement': k_fit_rule,
                      'time_solve': time_solve,
                      'time_cg': time_cg,
                      'time_preconditioner': time_preconditioner,
                      'optimal_experimental_k': optimal_measured_k,
                      'k_interpolated': k_measured_interpolate
                      }

    # relative cost of preconditioner vs iterative cg solve
    relative_costs = interp_time_cg / interp_time_precon
    arg_relative_2 = (np.abs(relative_costs - 2)).argmin()
    ratio2_k = k_measured_interpolate[arg_relative_2]
    relative_computational_ratio2 = interp_time_solve[arg_relative_2] / minimal_time_solve
    dict_optimal_k.update({'ratio2_k': ratio2_k, 'ratio2_factor': relative_computational_ratio2})

    # k_test_interpolate = np.linspace(1, n_kernel, 10000, dtype=np.int)
    for dataset_name in ['default', dict_molecule['dataset_name']]:
        slope, k_unity, prefactor = get_params(dataset_name=dataset_name)
        fixed_rule_of_thumb_fn = partial(rule_of_thumb_fn, n_kernel_rule=n_kernel, slope=slope, k_unity=k_unity,
                                         prefactor=prefactor)
        predicted_computational_costs = fixed_rule_of_thumb_fn(k_column=k_measured_interpolate * 1.)
        optimal_k = k_measured_interpolate[predicted_computational_costs.argmin()]
        relative_computational_ruleofthumb = interp_time_solve[predicted_computational_costs.argmin()] / minimal_time_solve
        if dataset_name == 'default':
            dict_optimal_k.update({f'rule_of_thumb_k_default': optimal_k,
                                   'rule_of_thumb_factor_default': relative_computational_ruleofthumb})
        else:
            dict_optimal_k.update({f'rule_of_thumb_k_specific': optimal_k,
                                   'rule_of_thumb_factor_specific': relative_computational_ruleofthumb})


    smallest_k = k_fit_rule.min()
    relative_computationl_smallest_k = time_solve[k_fit_rule.argmin()]/minimal_time_solve
    dict_optimal_k.update({'smallest_k': smallest_k, 'smallest_factor': relative_computationl_smallest_k})

    # naive preconditioner size based on fixed percentage of columns
    p = 1 / 100
    argmin_naive = np.argmin(np.abs(k_measured_interpolate / n_kernel - p))
    # argmin_naive = np.argmin(np.abs(k_measured_interpolate - 1000))
    naive_k = k_measured_interpolate[argmin_naive]
    naive_relative_runtime = interp_time_solve[argmin_naive] / minimal_time_solve
    dict_optimal_k.update({'naive_k': naive_k, 'naive_factor': naive_relative_runtime})
    return dict_optimal_k


def measure_optimal_k(dict_train: Dict):
    name_dataset = dict_train["dataset_name"]
    n_datapoints = max(dict_train['list_n_datapoints'])
    n_kernel = dict_train[f'{n_datapoints}_K.shape'][0]

    dict_k = calculate_optimal_precon_k(dict_molecule=dict_train, n_datapoints=n_datapoints)

    platform = dict_train['platform']
    print(f'---------------------------------------------------------------------------------------------------------\n'
          f'{str(name_dataset).capitalize()}, n = {n_kernel}\n'
          f'optimal measured:      time = {dict_k["time_solve"].min() / 60:.1f} min, k = {dict_k["optimal_experimental_k"]:.0f}\n'
          f'minimal k:             r = {dict_k["smallest_factor"]:.2f}  k = {dict_k["smallest_k"]:.0f}\n'
          f'RuleofThumb (default): r = {dict_k["rule_of_thumb_factor_default"]:.2f}  k = {dict_k["rule_of_thumb_k_default"]:}\n'
          f'RuleofThumb (specifc): r = {dict_k["rule_of_thumb_factor_specific"]:.2f}  k = {dict_k["rule_of_thumb_k_specific"]:}\n'
          f'ratio precon/cg = 2:   r = {dict_k["ratio2_factor"]:.2f}  k = {dict_k["ratio2_k"]}\n'
          f'Naive fixed size:      r = {dict_k["naive_factor"]:.2f}  k = {dict_k["naive_k"]}\n'
          # f'Node: {platform.node}, machine: {platform.machine}\n'
          f'---------------------------------------------------------------------------------------------------------\n'
          )


def get_cpugpu_info(all_molecules_runs: List[Dict]) -> pd.DataFrame:
    from collections import Counter
    def get_kernel_size(dict_molecule: Dict) -> int:
        list_shape_key = [key for key in dict_molecule.keys() if key.endswith('.shape')]
        assert len(list_shape_key) == 1, 'Only exactly a single measurements allowed.'
        shape_key = list_shape_key[0]
        return dict_molecule[shape_key][0]

    list_n = [round(get_kernel_size(dict_molecule), -2) for dict_molecule in all_molecules_runs]
    n = Counter(list_n).most_common(1)[0][0]

    def get_molecule_name(dict_molecule: Dict) -> str:
        name = str(dict_molecule['dataset_name'])
        return map_dataset_name_to_molecule(dataset_name=name)

    dict_node = {get_molecule_name(dict_molecule): dict_molecule['platform'].node for dict_molecule in all_molecules_runs}
    dict_gpu = {molecule: convert_node_to_gpu(dict_node[molecule]) for molecule in dict_node}
    dict_cpu = {molecule: convert_node_to_cpu(dict_node[molecule]) for molecule in dict_node}

    df_cpugpu = pd.DataFrame([dict_node, dict_cpu, dict_gpu], index=['node', 'CPU', 'GPU'])
    df_cpugpu = pd.DataFrame([dict_cpu, dict_gpu], index=['CPU', 'GPU'])
    sort_molecules = ['Ethanol', 'Uracil', 'Toluene', 'Aspirin', 'Azobenzene', 'Catcher', 'Nanotube']
    df_cpugpu = df_cpugpu[sort_molecules]
    multi_index = pd.MultiIndex.from_tuples([(n, name) for name in df_cpugpu.index])
    df_cpugpu_multiindex = pd.DataFrame(df_cpugpu.to_numpy(), index=multi_index, columns=df_cpugpu.columns)
    # df_cpugpu['n'] = [n for _ in range(3)]
    # df_cpugpu.index.name = 'info'
    return df_cpugpu_multiindex


def rule_of_thumb(dict_train: Dict, dict_test: Dict):

    n_datapoints_fit_rule = max(dict_train['list_n_datapoints'])
    n_kernel = dict_train[f'{n_datapoints_fit_rule}_K.shape'][0]
    k_fit_rule = dict_train[f'{n_datapoints_fit_rule}_lev_random_percentage'] * n_kernel
    # cg_steps_normalized = dict_train[f'{n_datapoints_fit_rule}_lev_random_cgsteps'] / n_kernel
    time_solve = dict_train[f'{n_datapoints_fit_rule}_lev_random_total_time_solve']

    def f_rule_of_thumb(k_column: np.ndarray, slope: float, prefactor: float, speed_up_cholesky: float,
                        n_kernel_rule: int):
        slope = 1
        speed_up_cholesky = 1.e-2
        prefactor = 1.e3
        return prefactor * (k_column ** (-slope) + speed_up_cholesky * (k_column / n_kernel_rule) ** 2)

    # a = partial(rule_of_thumb, n_kernel_rule=n_kernel)
    params_rule_of_thumb, pcov = curve_fit(f=partial(f_rule_of_thumb, n_kernel_rule=n_kernel),
                                           xdata=k_fit_rule, ydata=time_solve / time_solve.min(),
                                           # p0=[1, 1, 1],
                                           bounds=(0.0001, np.inf))  # bounds=([-10, 0, 0], [0, 1500, 1]),

    # params_rule_of_thumb = [0.72, 105, 0.013]
    print(f'fitted params rule of thumb: \n{params_rule_of_thumb}')

    dataset_name = str(dict_train['dataset_name'])
    fig = plt.figure(f'rule of thumb - {dataset_name}', figsize=(6, 2.5))
    # for i, n_datapoints in enumerate(dict_test['list_n_datapoints'][:]):
    for index, dict_plot in enumerate([dict_train, dict_test]):
        ax = plt.subplot(2, 1, 1+index)

        if index == 0:
            n_datapoints = n_datapoints_fit_rule
        else:
            n_datapoints = max(dict_test['list_n_datapoints'])


        plt.xlabel('percentage of columns')
        n_kernel = dict_plot[f'{n_datapoints}_K.shape'][0]
        percentage_preconditioning = dict_plot[f'{n_datapoints}_lev_random_percentage']
        k_data_test = percentage_preconditioning * n_kernel

        cg_steps_normalized = dict_plot[f'{n_datapoints}_lev_random_cgsteps'] / n_kernel
        time_solve = dict_plot[f'{n_datapoints}_lev_random_total_time_solve']
        time_preconditioner = dict_plot[f'{n_datapoints}_lev_random_total_time_preconditioner']
        time_cg = dict_plot[f'{n_datapoints}_lev_random_total_time_cg']


        k_test = np.arange(min(percentage_preconditioning * n_kernel) * 0.9,
                           max(percentage_preconditioning * n_kernel) * 1.1, 1)
        time_rule_of_thumb = f_rule_of_thumb(k_test, *params_rule_of_thumb, n_kernel_rule=n_kernel)

        # n_kernel = 1
        plt.plot(percentage_preconditioning*100, time_solve / time_solve.min(), '--x', label=f'measured', color='C0')
        plt.plot(k_test/n_kernel*100, time_rule_of_thumb / time_rule_of_thumb.min(), '-', label=f'predicted', color='C1')

        plt.plot(percentage_preconditioning * 100, time_preconditioner/time_cg, '--x', label=f'precon/cg', color='C4')
        plt.ylim(0, 1.1*max(time_solve / time_solve.min()))
        # plt.plot(percentage_preconditioning * 100, time_cg / time_solve, '--x', label=f'cg/total',
        #          color='C5')

        if index == 0:
            plt.ylabel('runtime [a.u.]                        ', loc='top')
            plt.legend(loc='upper right')

        argmin = np.argmin(time_rule_of_thumb)
        dataset_name = str(dict_plot['dataset_name'])
        text_str = f'{dataset_name} ({n_datapoints})\nOptimal k = {k_test[argmin]:.0f}, {k_test[argmin]/n_kernel:.1%}'
        at = AnchoredText(text_str, prop=dict(size=10), frameon=True, loc='upper center', pad=0.5)
        at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
        ax.add_artist(at)

        # ax2 = ax.twinx()
        # ax2.plot(percentage_preconditioning, cg_steps_normalized, '-.x', label=f'cg_steps: {n_datapoints}', color='C4')
        # ax2.set_ylabel('cg steps')
        plt.ylim(0.25, 3.5)
        plt.tight_layout(pad=0.1)


def computational_cost_molecule(dict_molecule: Dict):

    n_datapoints = max(dict_molecule['list_n_datapoints'])
    n_kernel = dict_molecule[f'{n_datapoints}_K.shape'][0]
    dataset_name = str(dict_molecule['dataset_name'])

    k_fit_rule = dict_molecule[f'{n_datapoints}_lev_random_percentage'] * n_kernel
    # cg_steps_normalized = dict_train[f'{n_datapoints_fit_rule}_lev_random_cgsteps'] / n_kernel

    # for i, n_datapoints in enumerate(dict_test['list_n_datapoints'][:]):

    n_kernel = dict_molecule[f'{n_datapoints}_K.shape'][0]
    percentage_preconditioning = dict_molecule[f'{n_datapoints}_lev_random_percentage']
    k_data_test = percentage_preconditioning * n_kernel

    cg_steps_normalized = dict_molecule[f'{n_datapoints}_lev_random_cgsteps'] / n_kernel
    time_solve_s = dict_molecule[f'{n_datapoints}_lev_random_total_time_solve']
    time_preconditioner = dict_molecule[f'{n_datapoints}_lev_random_total_time_preconditioner']
    time_cg = dict_molecule[f'{n_datapoints}_lev_random_total_time_cg']

    fig = plt.figure(f'computational_costs_{dataset_name}_{n_kernel}', figsize=(6, 2.5))
    time_solve_min = time_solve_s/60
    plt.plot(percentage_preconditioning*100, time_solve_min, '-x', label=f'Measured runtime', color='C0', markersize=3)

    ax = plt.gca()
    ax.axhline(y=time_solve_min.min(), linewidth=1, color=color_vertical_cost_line)

    dict_k = calculate_optimal_precon_k(dict_molecule=dict_molecule, n_datapoints=n_datapoints)
    ax.axvline(dict_k["optimal_experimental_k"]/n_kernel*100, color='green', label='Minimum')
    ax.axvline(dict_k["rule_of_thumb_k_default"] / n_kernel * 100, color='red', label='Rule of Thumb (default)')
    ax.axvline(dict_k["rule_of_thumb_k_specific"] / n_kernel * 100, color='blue', label='Rule of Thumb (specific)')
    ax.axvline(dict_k["ratio2_k"] / n_kernel * 100, color='orange', label=r'ratio 2')

    plt.xlabel('percentage of columns')
    plt.ylabel('runtime [min]')
    plt.legend()
    plt.tight_layout(pad=0.1)


def visualize_ridge_leverage_scores(dict_ridge: Dict):
    figsize = (4, 2.2)
    plt.figure('ridge leverage scores', figsize=figsize)
    ax1 = plt.subplot(2, 1, 1)

    # for lam in np.geomspace(1e-6, 1e-13, 5):
    for lam_temp in dict_ridge['list_lambda']:
        tau = dict_ridge[f'tau_{lam_temp}']

        # ax1.plot(tau, label=f'ridge: {lam:.0e}')
        ax1.plot(np.sort(tau), label=f'{lam_temp:.0e}')

    ax1.legend(ncol=4, handlelength=1, columnspacing=1)
    ax1.set_ylim([-0.1, 1.1])
    # ax1.semilogy()

    ax2 = plt.subplot(2, 1, 2)

    # calculate ridge leverage scores
    lam = 1e-10
    # K_lam = K.copy() @ K.T.copy() + lam * np.eye(n)
    # temp = sp.linalg.solve(K_lam, K)
    tau =dict_ridge[f'tau_{lam}']

    molecule_size = dict_ridge['molecule_size']
    n_train = dict_ridge['n_train']
    z = np.array([6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1])

    # sort leverage scores periodically wrt to training points
    lev_scores_sorted_matrix = np.zeros((molecule_size, n_train))
    for i in range(n_train):
        lev_scores_sorted_matrix[:, i] = tau[i * molecule_size:(i + 1) * molecule_size]

    index_atoms = np.arange(20)
    index_atoms = np.argsort(lev_scores_sorted_matrix.sum(axis=0))[::-int(n_train/20)]       # sort data samples and select uniformly from small to large average lev_score
    mat = lev_scores_sorted_matrix[:, index_atoms]
    # mat = lev_scores_sorted_matrix[:, :20]
    # mat = np.sort(lev_scores_sorted_matrix, axis=1)[:, :20]
    vmax = np.abs(mat).max()
    cbar = ax2.imshow(mat.T, cmap="YlOrRd", vmax=vmax, vmin=0)
    ax2.set_ylabel('data points')
    ax2.set_yticks(np.arange(0, 20., 5))
    ax2.set_xlabel('atomic dimension')
    xticks_pos = np.arange(0, molecule_size, 3) + 1
    plt.xticks(xticks_pos, map_z_to_atom_type(z))
    ax2.tick_params(left=False,
                    bottom=False,
                    labelleft=True,
                    labelbottom=True)
    plt.colorbar(cbar)

    plt.tight_layout(pad=0.1)


def bar_plot_rule_of_thumb(path_to_data: str):
    """Plots all experiments into a single barplot. containing all molecules"""
    df_data_table = pd.read_csv(path_to_data, delimiter=';')
    df_data_table.drop('Unnamed: 0', axis=1, inplace=True)
    # df_data_table.RoT_specific_runtime = df_data_table.RoT_specific_runtime.apply(func=lambda s: s.replace(',', '.'))
    # df_data_table.ratio_2_runtime = df_data_table.ratio_2_runtime.apply(func=lambda s: s.replace(',', '.'))
    # df_data_table['optimal_runtime_min'] = df_data_table.optimal_runtime.apply(func=lambda s: s.strip('min'))
    # df_data_table['n'] = df_data_table['Unnamed: 4'].apply(lambda s: float(s.split('*')[0]) * 10 ** int(s.split('^')[1]))
    #
    # df_data_table.drop(['Unnamed: 4', 'optimal_runtime'], axis=1, inplace=True)
    # df_data_table.to_csv('/Users/sbluecher/Documents/University/git/cholesky/project/figures/paper_plots/rule_of_thumb2.csv')
    # plt.figure()
    df_data_table['optimal_runtime_normalized'] = pd.DataFrame(np.ones(len(df_data_table)))
    # df_data_table['description'] = pd.DataFrame(np.ones(len(df_data_table)))
    # df_data_table.get(['ones', 'RoT_default_runtime', 'RoT_specific_runtime']).plot.bar()

    fig, ax_np = plt.subplots(2, 1, num='rule_of_thumb_experiment')
    ax = ax_np[0]

    select_matrix_size = [75000, 158000, 500000]
    df_data_table = df_data_table[df_data_table['n'].isin(select_matrix_size)]
    pivot_table = pd.pivot_table(df_data_table, values=['RoT_default_runtime', 'RoT_specific_runtime'],
                                 index=['Molecule'], columns='n')
    # all_columns = pivot_table.columns
    # mask_target_matrix_size = all_columns.isin(select_matrix_size, level=1)
    # pivot_table = pivot_table[all_columns[mask_target_matrix_size]]
    colors = ['#fcc669', '#f9a106', '#b27304', '#67c8fe', '#01a4fe', '#0175b5']
    legends = [*[f'Default: {n}' for n in select_matrix_size],
               *[f'Specific: {n}' for n in select_matrix_size]]
    bool_sort_n = False

    def sort_colors_legends(pivot_table, colors, legends):
        pivot_table.columns = pivot_table.columns.swaplevel(0, 1)
        pivot_table.sort_index(axis=1, level=0, inplace=True)
        sort_index = [0, 3, 1, 4, 2, 5]
        colors = [colors[i] for i in sort_index]
        legends = [legends[i] for i in sort_index]
        return pivot_table, colors, legends

    if bool_sort_n is True:
        pivot_table, colors, legends = sort_colors_legends(pivot_table, colors, legends)


    # colors created with
    # https://mdigi.tools/color-shades/#de8f05
    pivot_table.plot.bar(ax=ax, color=colors, width=0.85, alpha=1)

    # ax = df_data_table.plot.bar(ax=ax, x='Molecule', y=['RoT_default_runtime', 'RoT_specific_runtime'],
    #                             color=['C1', 'C3'], width=0.85, figsize=(7, 3.2), legend=None)
    ax.legend(legends, ncol=2,
              title=r'Rule of Thumb ($n$)')
    ax.set_xlabel(None)
    ax.set_ylabel(r'Relative runtime $\left[100 \, \% \right]$')
    ax.axhline(1, c='C2')

    ax.set_ylim(0.75, 2.9)
    ax.set_yticks(np.array([1, 1.5, 2.0, 2.5]))
    ax.tick_params(left=True,
                    bottom=False,
                    labelleft=True,
                    labelbottom=False)


    # df_data_table['RoT_default_runtime_min'] = df_data_table['RoT_default_runtime'] * df_data_table['optimal_runtime_min']
    # df_data_table['RoT_specific_runtime_min'] = df_data_table['RoT_specific_runtime'] * df_data_table['optimal_runtime_min']
    # ax = df_data_table.plot.bar(x='Molecule',
    #                             y=['optimal_runtime_min', 'RoT_default_runtime_min', 'RoT_specific_runtime_min'],
    #                             color=['C2', 'C1', 'C3'], width=0.85, legend=None)


    ax = ax_np[1]

    df_data_table['RoT_default_columns_relative'] = df_data_table['RoT_default_columns'] / df_data_table['optimal_columns']
    df_data_table['RoT_specific_columns_relative'] = df_data_table['RoT_specific_columns'] / df_data_table['optimal_columns']
    ax.set_ylabel(r'Relative size $k$ $\left[100 \, \% \right]$')
    pivot_table = pd.pivot_table(df_data_table,
                                 values=['RoT_default_columns_relative', 'RoT_specific_columns_relative'],
                                 index=['Molecule'], columns='n')
    pivot_table.plot.bar(ax=ax, color=colors, width=0.85,
                         alpha=1, legend=None)

    if bool_sort_n is True:
        pivot_table, colors, legends = sort_colors_legends(pivot_table, colors, legends)

    # ax = df_data_table.plot.bar(ax=ax, x='Molecule',
    #                             y=['RoT_default_columns_relative', 'RoT_specific_columns_relative'],
    #                             color=['C1', 'C3'], width=0.85)
    # plt.legend(['Rule of Thumb: default', 'Rule of Thumb: optimized'])
    ax.axhline(1, c='C2')
    ax.legend(['Optimum experiment'])
    ax.tick_params(left=True,
                   bottom=False,
                   labelleft=True,
                   labelbottom=True)
    ax.semilogy()
    # plt.yticks(ticks=np.array([0.5, 1, 4]), labels=['50%', '100%', '400%'])
    ax.set_xlabel(None)
    ax.tick_params(rotation=0)
    plt.tight_layout(pad=0.1)           # change overall padding to be tight
    plt.subplots_adjust(hspace=0)       # change padding between all subplots, call after tight_layout()


def bar_plot_rule_of_thumb_single_experiment(path_to_data: str, n: int):
    """Plots a single experiments (n) into a single barplot. containing all molecules"""
    df_data_table = pd.read_csv(path_to_data, delimiter=';')
    df_data_table.drop('Unnamed: 0', axis=1, inplace=True)

    df_data_table['optimal_runtime_normalized'] = pd.DataFrame(np.ones(len(df_data_table)))

    fig = plt.figure(figsize=(9, 2.), num=f'rule_of_thumb_experiment_{n}')
    ax = fig.subplots()

    select_matrix_size = [n]
    df_data_table = df_data_table[df_data_table['n'].isin(select_matrix_size)]
    pivot_table = pd.pivot_table(df_data_table, values=['naive_1per_runtime', 'RoT_default_runtime', 'RoT_specific_runtime'],
                                 index=['Molecule'], columns='n')
    pivot_table = pivot_table[pivot_table.columns[[2, 0, 1]]]   # manual sort: alphabetic -> (naive, default, specific)
    molecule_sorted = ['Ethanol', 'Uracil', 'Toluene', 'Aspirin', 'Azobenzene', 'Catcher', 'Nanotube']
    pivot_table = pivot_table.loc[molecule_sorted]

    colors = ['C7', 'C9', 'C0']         # based on sns.color_palette('colorblind')

    legends = [r'$1\%$-baseline', 'Default', 'Specific']
    bool_sort_n = False

    def sort_colors_legends(pivot_table, colors, legends):
        pivot_table.columns = pivot_table.columns.swaplevel(0, 1)
        pivot_table.sort_index(axis=1, level=0, inplace=True)
        sort_index = [0, 3, 1, 4, 2, 5]
        colors = [colors[i] for i in sort_index]
        legends = [legends[i] for i in sort_index]
        return pivot_table, colors, legends

    if bool_sort_n is True:
        pivot_table, colors, legends = sort_colors_legends(pivot_table, colors, legends)


    # colors created with
    # https://mdigi.tools/color-shades/#de8f05
    pivot_table.plot.bar(ax=ax, color=colors, width=0.85, alpha=1)

    # ax = df_data_table.plot.bar(ax=ax, x='Molecule', y=['RoT_default_runtime', 'RoT_specific_runtime'],
    #                             color=['C1', 'C3'], width=0.85, figsize=(7, 3.2), legend=None)
    l1 = ax.legend(legends, title=r'Rule of Thumb ($n = $' + f'{n})', ncol=3)

    ax.set_xlabel(None)
    ax.set_ylabel(r'Relative runtime $\left[100 \, \% \right]$')
    p1 = ax.axhline(1, c='C3')

    # # This removes l1 from the axes.
    # plt.legend([p1], ['Experimental minimum'], loc=(0.757, 0.595))
    if n == 75000:
        loc = 'upper left'
        loc = (0.82, 0.6)  # position relative to origin (x, y)
    elif n == 158000:
        loc = 'upper right'
        loc = (0.0065, 0.6)  # position relative to origin (x, y)
    elif n == 500000:
        loc = 'center right'
        loc = (0.819, 0.6)    # position relative to origin (x, y)
    else:
        loc = 'upper left'
    plt.legend([p1], ['Experimental minimum'], loc=loc)
    # Add l1 as a separate artist to the axes
    ax.add_artist(l1)

    ax.set_ylim(0.75, 1.*ax.get_ylim()[1])
    # ax.set_yticks(np.array([1, 1.5, 2.0, 2.5]))
    # ax.tick_params(left=True,
    #                 bottom=False,
    #                 labelleft=True,
    #                 labelbottom=False)


    # df_data_table['RoT_default_runtime_min'] = df_data_table['RoT_default_runtime'] * df_data_table['optimal_runtime_min']
    # df_data_table['RoT_specific_runtime_min'] = df_data_table['RoT_specific_runtime'] * df_data_table['optimal_runtime_min']
    # ax = df_data_table.plot.bar(x='Molecule',
    #                             y=['optimal_runtime_min', 'RoT_default_runtime_min', 'RoT_specific_runtime_min'],
    #                             color=['C2', 'C1', 'C3'], width=0.85, legend=None)


    # ax = ax_np[1]
    #
    # df_data_table['RoT_default_columns_relative'] = df_data_table['RoT_default_columns'] / df_data_table['optimal_columns']
    # df_data_table['RoT_specific_columns_relative'] = df_data_table['RoT_specific_columns'] / df_data_table['optimal_columns']
    # df_data_table['naive_1per_columns_relative'] = df_data_table['naive_1per_columns'] / df_data_table['optimal_columns']
    #
    # ax.set_ylabel(r'Relative size $k$ $\left[100 \, \% \right]$')
    # pivot_table = pd.pivot_table(df_data_table,
    #                              values=['RoT_default_columns_relative', 'RoT_specific_columns_relative', 'naive_1per_columns_relative'],
    #                              index=['Molecule'], columns='n')
    # pivot_table = pivot_table[pivot_table.columns[[2, 0, 1]]]  # manual sort: alphabetic -> (naive, default, specific)
    # pivot_table = pivot_table.loc[molecule_sorted]
    #
    # pivot_table.plot.bar(ax=ax, color=colors, width=0.85,
    #                      alpha=1, legend=None)
    #
    # if bool_sort_n is True:
    #     pivot_table, colors, legends = sort_colors_legends(pivot_table, colors, legends)
    #
    # # ax = df_data_table.plot.bar(ax=ax, x='Molecule',
    # #                             y=['RoT_default_columns_relative', 'RoT_specific_columns_relative'],
    # #                             color=['C1', 'C3'], width=0.85)
    # # plt.legend(['Rule of Thumb: default', 'Rule of Thumb: optimized'])
    # ax.axhline(1, c='C2')
    # ax.legend(['Optimum experiment'])
    # ax.tick_params(left=True,
    #                bottom=False,
    #                labelleft=True,
    #                labelbottom=True)
    # ax.semilogy()
    # plt.yticks(ticks=np.array([0.5, 1, 4]), labels=['50%', '100%', '400%'])
    ax.set_xlabel(None)
    ax.tick_params(rotation=0)

    df_data_table.index = df_data_table['Molecule']
    list_optimal_runtime = [df_data_table.loc[molecule]['optimal_runtime_min'] for molecule in molecule_sorted]
    xtickslabels = [f'{molecule} \n({runtime:.0f}min)' for molecule, runtime in zip(molecule_sorted, list_optimal_runtime)]

    ax.set_xticklabels(xtickslabels)

    plt.tight_layout(pad=0.1)           # change overall padding to be tight
    plt.subplots_adjust(hspace=0)       # change padding between all subplots, call after tight_layout()


def rule_of_thumb(n: Union[np.ndarray, int], k_min: int, m: float) -> Union[np.ndarray, int]:
    res = (k_min**m * m * n**2 / 2) ** (1/(2 + m))
    if isinstance(n, int):
        res = int(np.floor(res))
    return res


def visualize_rule_of_thumb(relative_to_kernel_size=True):
    plt.figure('Rule of Thumb - theoretical comparison', figsize=(3.4, 2.5))
    n = np.arange(1_000, 10_000_000, 10**3)
    # k_default = 0.01 * n

    for p in [0.01]:
        k = p * n
        if relative_to_kernel_size:
            k *= 1/n
        plt.plot(n, k, label=f'Baseline: {p:.0%}')

    for dataset_name in ['default', 'ethanol', 'aspirin', 'aims_catcher', 'larger_aims_nanotube']:
        m, k_min, _ = get_params(dataset_name=dataset_name, old=False)
        k_RoT = rule_of_thumb(n=n, k_min=k_min, m=m)
        if relative_to_kernel_size:
            k_RoT *= 1/n
        plt.plot(n, k_RoT, label=map_dataset_name_to_molecule(dataset_name))

    plt.legend(ncol=2)
    if relative_to_kernel_size:
        plt.ylabel(r'relative preconditioner size $\frac{k}{n}$')
    else:
        plt.ylabel(r'preconditioner size $k$')
    plt.xlabel(r'kernel size $n$')
    plt.semilogx()
    plt.tight_layout(pad=0.1)


def plane_cg_steps_difference(ax: plt.Axes, dict_data: Dict, reference_label: str, label: str):
    x_reference, y_reference, _, _ = calculate_error(x=dict_data[f'{reference_label}_percentage'],
                                                  y=dict_data[f'{reference_label}_cgsteps'],
                                                  return_error=False)
    f_reference = interp1d(x_reference, y_reference, kind='linear')
    x_target, y_target, _, n_min = calculate_error(x=dict_data[f'{label}_percentage'],
                                            y=dict_data[f'{label}_cgsteps'],
                                            return_error=False)
    print(f'molecule: {dict_data["dataset_name"]}, preconditioner: {label} -> {n_min}')
    f_target = interp1d(x_target, y_target, kind='linear')

    x_range = np.linspace(max(x_target.min(), x_reference.min()), min(x_target.max(), x_reference.max()), 1000)
    ax.plot(x_range, f_target(x_range) - f_reference(x_range), c=map_dict_label_to_color(label),
             label=map_dict_label_to_paper(label))


def cg_steps_difference(dict_data: Dict, reference_label: str):
    all_labels = ['eigvec_precon', 'lev_random', 'cholesky', 'random_scores', 'lev_scores', 'inverse_lev',
                  'truncated_cholesky', 'truncated_cholesky_custom', 'rank_k_lev_scores']
    labels = [label for label in all_labels if f'{label}_percentage' and f'{label}_cgsteps' in dict_data]
    labels.remove(reference_label)

    n_kernel = dict_data['K.shape'][0]  # size kernel matrix
    dataset = str(dict_data["dataset_name"])

    fig = plt.figure(f'{dataset}, n={n_kernel}, difference to baseline: {map_dict_label_to_paper(reference_label)}')
    ax = fig.subplots()
    for label in labels:
        plane_cg_steps_difference(ax=ax, dict_data=dict_data, reference_label=reference_label, label=label)

    plt.semilogy()
    plt.legend()
    plt.tight_layout(pad=0.1)


def cg_steps_difference_all(list_dict_data: List[Dict], reference_label: str):
    list_dict_data.sort(key=lambda d: d['n_datapoints'], reverse=True)
    all_labels = ['eigvec_precon', 'lev_random', 'cholesky', 'random_scores', 'lev_scores', 'inverse_lev',
                  'truncated_cholesky', 'truncated_cholesky_custom', 'rank_k_lev_scores']
    n_methods = len(list_dict_data)
    figsize = (8, 1.2)
    figsize = (figsize[0], n_methods * figsize[1])

    fig = plt.figure(f'n={list_dict_data[0]["K.shape"][0]}, difference to baseline: {map_dict_label_to_paper(reference_label)}',
                     figsize=figsize)
    axs = fig.subplots(n_methods, 1, sharex=True)
    for index, [ax, dict_data] in enumerate(zip(axs, list_dict_data)):
        n_kernel = dict_data['K.shape'][0]  # size kernel matrix
        dataset = str(dict_data["dataset_name"])
        labels = [label for label in all_labels if f'{label}_percentage' and f'{label}_cgsteps' in dict_data]
        labels.remove(reference_label)

        for label in labels:
            plane_cg_steps_difference(ax=ax, dict_data=dict_data, reference_label=reference_label, label=label)
        ax.semilogy()
        text_str = f'{map_dataset_name_to_molecule(dataset)}'
        at = AnchoredText(text_str, prop=dict(size=10), frameon=True, loc='upper right', pad=0.5)
        at.patch.set_boxstyle("round, pad=0., rounding_size=0.3")
        ax.add_artist(at)
        if index == n_methods-1:
            # plt.legend(ncol=2, loc=(0.585, 0.285))
            plt.legend(ncol=n_methods, loc='upper center')
            ax.set_xlabel(r'percentage of columns $\frac{k}{n}$')
        elif index == int(n_methods / 2):
            ax.set_ylabel('Additional suboptimal steps: $\#_{method} - \#_{SVD}$')
        else:
            ax.tick_params(left=True,
                           bottom=False,
                           labelleft=True,
                           labelbottom=False)

    plt.tight_layout(pad=0.1)           # change overall padding to be tight
    plt.subplots_adjust(hspace=0)       # change padding between all subplots, call after tight_layout()