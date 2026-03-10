import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from samplers import sample_gue
from densities import semicircle_density

def wigner_simulation(ns=[100, 500, 1000, 2000], seed=None, savepath=None):
    """
    Simulate the Wigner semicircle distribution by sampling GUE matrices
    of the form (1/sqrt(n))H where H is Hermitian.
    """
    rng = np.random.default_rng(seed)

    k = len(ns)

    # Choose grid automatically (as square as possible)
    n_cols = math.ceil(math.sqrt(k))
    n_rows = math.ceil(k / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.array(axes).reshape(-1)  # flatten in case of 2D grid

    for i, n in tqdm(list(enumerate(ns)), total=k):
        H = sample_gue(n, rng=rng)
        eigs = np.linalg.eigvalsh(H / np.sqrt(n))

        xs = np.linspace(-2.2, 2.2, 600)

        ax = axes[i]
        ax.hist(eigs, bins='auto', density=True, alpha=0.7)
        ax.plot(xs, semicircle_density(xs), linewidth=2, label='Semicircle PDF')
        ax.set_xlim(-2.2, 2.2)
        ax.set_title(f"n={n}")
        ax.legend()

    # Hide unused subplots if grid > k
    for j in range(k, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(r"Wigner semicircle law (GUE): eigenvalues of $\frac{1}{\sqrt{n}}H$")
    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        print(f"Saved plot to {savepath}")
    

if __name__ == "__main__":
    wigner_simulation(ns=[30, 100, 500, 2000], savepath="./plots/wigner_simulation.png")