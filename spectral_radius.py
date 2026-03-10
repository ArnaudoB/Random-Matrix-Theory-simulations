import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from densities import gumbel_density

def fluctuations_spectral_radius_ginibre(
    ns=(200, 500, 1000, 1500),
    N_exp=10000,
    seed=None,
    batch_size=1000,
    savepath="./plots/fluctuations_spectral_radius_ginibre.png",
):
    """
    Complex Ginibre (via Kostlan): the unordered squared moduli are distributed as
    independent Gamma(k,1), k=1..n. Hence the spectral radius can be simulated
    without eigenvalue computations. We then apply Rider's normalization and
    compare to the Gumbel law.
    """
    rng = np.random.default_rng(seed)

    ns = list(ns)
    if min(ns) < 200:
        raise ValueError("For the Rider-type normalization to be valid, we need n >= 200.")

    Xns = np.zeros((len(ns), N_exp), dtype=float)

    for i, n in tqdm(list(enumerate(ns)), desc="Simulating spectral radius fluctuations"):
        # Precompute constants for this n
        gamma_n = np.log(n / (2 * np.pi)) - 2 * np.log(np.log(n))
        shift = 1.0 + np.sqrt(gamma_n / (4.0 * n))
        scale = np.sqrt(4.0 * n * gamma_n)

        # Shapes for Gamma(k,1), broadcasted across rows in a batch
        shapes = np.arange(1, n + 1, dtype=float)[None, :]  # (1, n)

        # Batch loop
        start = 0
        while start < N_exp:
            end = min(start + batch_size, N_exp)
            b = end - start

            # Sample Gamma variables for this batch: (b, n)
            gammas = rng.gamma(shape=shapes, scale=1.0, size=(b, n))

            # Spectral radius proxy per experiment (max over k=1..n)
            max_gamma = gammas.max(axis=1)     # (b,)
            radii = np.sqrt(max_gamma / n)     # (b,)

            # Rider normalization
            Xns[i, start:end] = scale * (radii - shift)

            start = end

    # Plotting
    k = len(ns)
    n_cols = math.ceil(math.sqrt(k))
    n_rows = math.ceil(k / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, n in enumerate(ns):
        ax = axes[i]
        ax.hist(Xns[i], bins=80, density=True, alpha=0.6, label="Empirical")

        xs = np.linspace(-2, 6, 400)
        ax.plot(xs, gumbel_density(xs), label="Gumbel PDF")

        ax.set_title(f"Fluctuations (n={n})")
        ax.legend()

    for j in range(k, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Rescaled spectral radius fluctuations vs. Gumbel (Complex Ginibre)")
    fig.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
        print(f"Saved plot to {savepath}")

    return Xns


if __name__ == "__main__":
    fluctuations_spectral_radius_ginibre(
        ns=[200, 1000, 10000, 200000],
        N_exp=10000,
        seed=42,
        savepath="./plots/fluctuations_spectral_radius_ginibre.png",
    )