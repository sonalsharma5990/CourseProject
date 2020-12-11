"""Creates output plots."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def plot_causal_purity(
        avg_causal,
        avg_purity,
        iterations,
        outfile,
        label_prefix,
        yticks_causal=None,
        yticks_purity=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax = axes[0]
    markers = ['d', 's', '^', 'x', '*']
    for i, (value, causal) in enumerate(avg_causal.items()):
        ax.plot(
            range(
                1,
                iterations + 1),
            causal,
            label=f'{label_prefix}{value}',
            marker=markers[i])
    ax.set_xlabel('Iteration', weight='bold')
    # ax.set_xlim(0, 6)
    # ax.set_xticks(range(1,6))
    # ax.set_xlim([1,5])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.grid()
    if yticks_causal is None:
        yticks_causal = np.arange(95.5, 100.5, .5)
    ax.set_yticks(yticks_causal)

    ax.set_ylabel('Average Causality Confidence', weight='bold')

    ax = axes[1]
    for i, (value, purity) in enumerate(avg_purity.items()):
        ax.plot(
            range(
                1,
                iterations + 1),
            purity,
            label=f'{label_prefix}{value}',
            marker=markers[i])
    ax.set_xlabel('Iteration', weight='bold')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.grid()
    ax.set_ylabel('Average Purity', weight='bold')
    # ax.set_xticks(range(1, 7))

    if yticks_purity is None:
        yticks_purity = np.arange(0, 125, 20)
    ax.set_yticks(yticks_purity)
    # plt.show()

    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)

    fig.tight_layout()
    plt.savefig(outfile)


def plot_for_mu(avg_causal_by_mu, avg_purity_by_mu, iterations=5):
    plot_causal_purity(
        avg_causal_by_mu,
        avg_purity_by_mu,
        iterations,
        'mu_graph.png',
        '$\mu$=')


def plot_for_tn(avg_causal_by_tn, avg_purity_by_tn, iterations=5):
    plot_causal_purity(
        avg_causal_by_tn,
        avg_purity_by_tn,
        iterations,
        'tn_graph.png',
        'tn=')


def plot_from_csv(filename, data_folder, plot_type='mu'):
    csv_data = pd.read_csv(filename).rename({'t_n': 'tn'}, axis='columns')
    # csv columns
    # mu,t_n,iteration,avg_significance,avg_purity
    out_file = ''
    label_prefix = ''
    if plot_type == 'mu':
        out_file = f'{data_folder}/mu_graph.png'
        label_prefix = r'$\mu$='
    elif plot_type == 'tn':
        out_file = f'{data_folder}/tn_graph.png'
        label_prefix = 'tn='
    else:
        ValueError('Invalid plot_type')

    avg_causality = {}
    avg_purity = {}

    iterations = 0

    keys = pd.unique(csv_data[plot_type])
    # print(keys)

    for k in keys:
        avg_causality[k] = csv_data[csv_data[plot_type]
                                    == k]['avg_significance'].to_numpy()
        avg_purity[k] = csv_data[csv_data[plot_type]
                                 == k]['avg_purity'].to_numpy()
        iterations = max(
            np.max(csv_data[csv_data[plot_type] == k]['iteration']), iterations)

    causality_min = csv_data['avg_significance'].min()
    causality_max = csv_data['avg_significance'].max()

    yticks_causal = np.arange(
        round(
            causality_min * 2) / 2 - .5,
        round(
            causality_max * 2) / 2 + .5,
        0.1)

    purity_min = csv_data['avg_purity'].min()
    purity_max = csv_data['avg_purity'].max()

    yticks_purity = np.arange(int(purity_min) - 5, int(purity_max) + 5, 2)

    # print(yticks_purity)
    # print(yticks_causal)

    plot_causal_purity(
        avg_causality,
        avg_purity,
        iterations,
        out_file,
        label_prefix,
        yticks_causal,
        yticks_purity)

    # avg_causal = df['avg_significance'].tolist()
    # avg_purity = df['avg_purity'].tolist()
