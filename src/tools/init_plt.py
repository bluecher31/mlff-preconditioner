import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.rcParams.update(plt.rcParamsDefault)
# plt.style.use('default')

#
# def calculate_fig_size_in_inches(fig_width_pt: float) -> Tuple[float, float]:
#     """
#
#     :param
#     fig_width_pt: use "\showthe\columnwidth" within a figure in your tex and search for it in the .log
#     :return:
#     tuple with fig_width, fig_height in inches using the golden mean as ratio
#     """
#     inches_per_pt = 1.0/72.27               # Convert pt to inch
#     golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
#     fig_width = fig_width_pt*inches_per_pt  # width in inches
#     fig_height = fig_width*golden_mean      # height in inches
#     # fig_size = [fig_width, fig_height]
#     return fig_width, fig_height
#
#
# def update_figsize(fig_width_pt: float):
#     fig_size = calculate_fig_size_in_inches(fig_width_pt)
#     plt.rcParams.update({'figure.figsize': fig_size})

def update_rcParams(fig_size = (6.88, 3.5), half_size_image=False):
    plt.style.use('seaborn-paper')

    params = {
        # 'text.usetex': True,
        # 'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        # "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        # "font.sans-serif": [],
        # "font.monospace": [],
        # 'axes.labelsize': 10,
        # 'font.size': 10,
        # 'legend.fontsize': 10,
        # 'legend.handlelength': 1.5,
        # 'legend.borderpad': 0.4,
        "lines.markeredgewidth": 1,             # needed to overwrite seaborn
        'axes.titlesize': 8,
        "axes.titlepad": 3,
        # 'axes.titlepad:': 1,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.figsize': fig_size,
        # 'savefig.directory': 'home/bluecher/Dokumente/Git',
        'savefig.format': 'pdf',
        'pgf.texsystem': 'pdflatex',
        # 'pgf.rcfonts': False,
        #  'lines.linewidth': 0.2,        # 0.2 for borders in ssc image plots
        # 'lines.markersize': 4.0,
        # 'axes.linewidth': 0.4,
        'axes.grid': False,
        'grid.color': '#D3D3D3',
        'grid.linestyle': '-',
        'grid.linewidth': 0.35,
        'grid.alpha': 0.8,
        'savefig.pad_inches': 0.01,
        # 'text.latex.preamble': ['\usepackage{palatino}']
        # 'text.latex.preamble': [],
        # 'axes.prop_cycle': plt.cycler('color', ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'])
        # 'axes.prop_cycle': plt.cycler('color', ['#9400d3', '#009e73', '#56b4e9', '#e69f00', '#e51e10', '#f0e442', '#0072b2'])
        # 'axes.prop_cycle': plt.cycler('color', ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'])          # old configuration
        'axes.prop_cycle': plt.cycler('color', ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161',
                                                '#FBAFE4', '#949494', '#ECE133', '#56B4E9'])      # seaborn colorblind
    }
    plt.rcParams.update(params)
    if half_size_image is True:
        fig_size = (3.29, 1.8)
        plt.rcParams.update({
            'axes.labelsize': 10,
            'font.size': 12,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.figsize': fig_size,
        })
