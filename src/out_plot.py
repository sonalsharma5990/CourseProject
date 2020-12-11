"""Creates output plots."""
import matplotlib.pyplot as plt
import numpy as np


def plot_causal_purity(avg_causal, avg_purity, iterations, outfile, label_prefix):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax = axes[0]
    for value, causal in avg_causal.items():
        ax.plot(causal, label=f'{label_prefix}{value}', marker='+')
    ax.set_xlabel('Iteration')
    # ax.set_xticks(np.arange(1,7))
    ax.set_yticks(np.arange(95.5, 100.5, .5))
    ax.set_ylabel('Average Causality Confidence')

    ax = axes[1]
    for value, purity in avg_purity.items():
        ax.plot(purity, label=f'{label_prefix}{value}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Purity')
    # ax.set_xticks(np.arange(1, 7))
    ax.set_yticks(np.arange(0, 125, 20))
    # plt.show()
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
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
