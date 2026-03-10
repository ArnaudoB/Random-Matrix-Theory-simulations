import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from samplers import sample_covariance
from densities import marchenko_pastur_density

def marchenko_pastur_simulation(ns=[500, 1000, 2000], cs=[0.2, 0.5, 0.8], seed=None, savepath=None):
    """
    Simulate the Marchenko-Pastur distribution by sampling covariance matrices of the form (1/n)XX^* where X is a p x n matrix with i.i.d. entries, 
    and p/n -> c as n -> infinity. For p, we will use p = int(c * n).
    """

    rng = np.random.default_rng(seed)

    # Choose grid automatically (as square as possible)
    n_cols = len(ns)
    n_rows = len(cs)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.array(axes).reshape(-1)  # flatten in case of 2D grid

    for i, n in tqdm(list(enumerate(ns)), total=len(ns)):
        for j, c in enumerate(cs):
            p = int(c * n)
            S = sample_covariance(n, p, rng=rng) / n
            eigs = np.linalg.eigvalsh(S)
            xs = np.linspace(-0.5, 4.5, 600)
            ax = axes[j * n_cols + i]
            bins = "auto"
            ax.hist(eigs, bins=bins, density=True, alpha=0.7)
            ax.plot(xs, marchenko_pastur_density(xs, c), linewidth=2, label='Marchenko-Pastur PDF')
            ax.set_xlim(-0.5, 4.5)
            ax.set_title(f"Simulation (n={n}, p={p}, c={c})")
            ax.legend()
    
    for j in range(len(cs) * len(ns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(r"Marchenko-Pastur law: eigenvalues of $\frac{1}{n}XX^*$")
    fig.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    marchenko_pastur_simulation(ns=[50, 250, 2000], cs=[0.2, 0.5, 0.8], savepath="./plots/marchenko_pastur_simulation.png")