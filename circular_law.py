import numpy as np
import matplotlib.pyplot as plt
from samplers import sample_ginibre, sample_bernoulli
from densities import circular_law_density
from tqdm import tqdm
import math


def circular_law_simulation(ns=[100, 500, 1000, 2000], seed=0, savepath=None, law='ginibre'):
    """
    Simulate the circular law by sampling matrices of the form (1/sqrt(n))G where G is an n x n matrix with i.i.d. entries and bounded moments
    """

    rng = np.random.default_rng(seed)

    k = len(ns)

    # Choose grid automatically (as square as possible)
    n_cols = math.ceil(math.sqrt(k))
    n_rows = math.ceil(k / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.array(axes).reshape(-1)  # flatten in case of 2D grid

    for i, n in tqdm(list(enumerate(ns)), total=len(ns)):

        if law == 'ginibre':
            G = sample_ginibre(n, rng=rng)
        elif law == 'bernoulli':
            G = sample_bernoulli(n, rng=rng)
        else:
            raise ValueError(f"Unsupported law: {law}. Supported options are 'ginibre' and 'bernoulli'.")

        eigs = np.linalg.eigvals(G / (np.sqrt(n)*np.std(G)))
        print(f"KS statistic for radius CDF convergence (n={n}, law={law}): {radial_cdf_error(eigs):.4f}")

        xs = np.linspace(-1.5, 1.5, 300)
        ys = np.linspace(-1.5, 1.5, 300)
        X, Y = np.meshgrid(xs, ys)
        Z = circular_law_density(X, Y)
        ax = axes[i]
        ax.scatter(eigs.real, eigs.imag, s=10, alpha=1.0, color='blue', label='Eigenvalues')
        ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        
        ax.set_title(f"{law.capitalize()} simulation (n={n})")

    for j in range(k, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(r"Circular law: eigenvalues of $\frac{1}{\sqrt{n}}G$")
    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()


def radial_cdf_error(eigs):
    """
    If the circular law holds, the empirical CDF of the radius of the eigenvalues should converge to F(r) = r^2 for r in [0,1]. 
    We can compute a distance metric like the Kolmogorov-Smirnov statistic to quantify the convergence.
    """
    r = np.abs(eigs)
    r_sorted = np.sort(r)
    n = len(r_sorted)
    empirical_cdf = np.arange(1, n+1) / n
    theoretical_cdf = r_sorted**2
    return np.max(np.abs(empirical_cdf - theoretical_cdf))