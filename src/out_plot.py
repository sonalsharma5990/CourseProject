"""Creates output plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_causal_purity(
        avg_causal,
        avg_purity,
        iterations,
        outfile,
        label_prefix):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    ax = axes[0]
    markers = ['d', 's', '^', 'x', '*']
    for i, (value, causal) in enumerate(avg_causal.items()):
        ax.plot(causal, label=f'{label_prefix}{value}', marker=markers[i])
    ax.set_xlabel('Iteration', weight='bold')
    ax.set_xticks(range(1,7))
    ax.yaxis.grid()
    ax.set_yticks(np.arange(95.5, 100.5, .5))
    ax.set_ylabel('Average Causality Confidence', weight='bold')

    ax = axes[1]
    for i, (value, purity) in enumerate(avg_purity.items()):
        ax.plot(purity, label=f'{label_prefix}{value}', marker=markers[i])
    ax.set_xlabel('Iteration', weight='bold')
    ax.yaxis.grid()
    ax.set_ylabel('Average Purity', weight='bold')
    ax.set_xticks(range(1, 7))
    ax.set_yticks(np.arange(0, 125, 20))
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

def plot_from_mu_csv(filename):
    csv_data = pd.read_csv(filename).todict()

    # mu,t_n,iteration,avg_significance,
    # avg_causal = df['avg_significance'].tolist()
    # avg_purity = df['avg_purity'].tolist()

