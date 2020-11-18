"""Creates output plots."""
import matplotlib.pyplot as plt


def plot_for_mu(avg_causal_by_mu, avg_purity_by_mu):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=90)
    ax = axes[0]
    for mu, causal in avg_causal_by_mu.items():
        ax.plot(causal)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Causality Confidence')

    ax = axes[1]
    for mu, purity in avg_purity_by_mu.items():
        ax.plot(purity)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Purity')
    # plt.show()
    plt.savefig('mu_graph.png')


def plot_for_tn(avg_causal_by_tn, avg_purity_by_tn):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=90)
    ax = axes[0]
    for mu, causal in avg_causal_by_tn.items():
        ax.plot(causal)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Causality Confidence')

    ax = axes[1]
    for mu, purity in avg_purity_by_tn.items():
        ax.plot(purity)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Purity')
    # plt.show()
    plt.savefig('tn_graph.png')
