import numpy as np
import matplotlib.pyplot as plt
from samplers import sample_ginibre, sample_bernoulli
from densities import circular_law_density
from tqdm import tqdm
import math
import os

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
        
        ax.set_title(f"Simulation (n={n})")

    for j in range(k, len(axes)):
        axes[j].set_visible(False)

    if law == 'ginibre':
        fig.suptitle(r"Circular law: eigenvalues of $\frac{1}{\sqrt{n}}G$ (Complex Ginibre)")
    elif law == 'bernoulli':
        fig.suptitle(r"Circular law: eigenvalues of $\frac{1}{\sqrt{n}}G$ (Bernoulli)")
    else:
        fig.suptitle(r"Circular law: eigenvalues of $\frac{1}{\sqrt{n}}G$")
    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        print(f"Saved plot to {savepath}")


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

def sample_sparse_matrix(n, alpha=0.5, rng=None, dist='normal'):
    if rng is None:
        rng = np.random.default_rng()
        
    rho = n**(-1 + alpha)
    mask = rng.random((n, n)) < rho
    
    if dist == 'normal':
        X = (rng.normal(0, 1, (n, n)) + 1j * rng.normal(0, 1, (n, n))) / np.sqrt(2)
    elif dist == 'bernoulli':
        X = (rng.choice([-1, 1], (n, n)) + 1j * rng.choice([-1, 1], (n, n))) / np.sqrt(2)
    else:
        raise ValueError("Unsupported base distribution")
        
    N_rho = mask * X
    return N_rho, rho

def circular_law_simulation_alphas(n=1000, alphas=[0.5, 0.2, 0.0], seed=42):
    
    rng = np.random.default_rng(seed)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, alpha in enumerate(alphas):
        print(f"Computing for alpha = {alpha}...")
        G, rho = sample_sparse_matrix(n, alpha=alpha, rng=rng)

        scaling_factor = 1.0 / np.sqrt(n * rho)
        scaled_G = G * scaling_factor
        eigs = np.linalg.eigvals(scaled_G)

        ax = axes[i]
        
        ax.set_aspect('equal', adjustable='box')
        
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='red', linestyle='--', linewidth=1.5, zorder=2)
        
        ax.scatter(eigs.real, eigs.imag, s=10, alpha=0.6, color='blue', zorder=3)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        actual_sparsity = 1.0 - (np.count_nonzero(G) / (n*n))
        ax.set_title(f"$\\alpha$={alpha} | $\\rho$={rho:.4f}\nZeros: {actual_sparsity*100:.2f}%")
        

    fig.suptitle(f"Evolution of the circular law for sparse matrices ($n={n}$)", fontsize=16, y=1.05)
    fig.tight_layout()

    
    savepath = os.path.join('plots', f'circular_law_evolution_n{n}.png')
    plt.savefig(savepath, bbox_inches='tight', dpi=300)
    print(f"\nFigure saved to: {savepath}")

    plt.show()

if __name__ == "__main__":
    
    circular_law_simulation_alphas(n=1000, alphas=[0.5, 0.2, 0.0])


if __name__ == "__main__":
    circular_law_simulation(ns=[10, 100, 500, 2000], seed=0, savepath="./plots/circular_law_ginibre.png", law='ginibre')
    circular_law_simulation(ns=[10, 100, 500, 2000], seed=0, savepath="./plots/circular_law_bernoulli.png", law='bernoulli')