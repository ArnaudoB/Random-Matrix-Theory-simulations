# Random Matrix Theory Simulations

Numerical simulations illustrating several classical results in **Random Matrix Theory (RMT)** showcased in the report *Expository notes on the circular law and the small singular value problem*.

This repository contains Python scripts used to simulate and visualize the asymptotic behavior of eigenvalues and spectral statistics of large random matrices.


## Overview

Random Matrix Theory studies the statistical behavior of eigenvalues of large matrices with random entries. Many remarkable universal phenomena emerge in the large dimension limit.

This repository implements simulations related to several fundamental results:

- **Wigner Semicircle Law**
- **Circular Law**
- **Spectral Radius Fluctuations**
- Other numerical experiments related to eigenvalue distributions.

All simulations rely on Monte Carlo experiments and visualization of empirical distributions.


## Implemented Experiments

### Wigner Semicircle Law

For large Hermitian random matrices (Wigner matrices), the empirical spectral distribution converges to the **semicircle distribution**

$$
\rho(x) = \frac{1}{2\pi}\sqrt{4 - x^2}, \quad |x| \le 2
$$

The scripts simulate large symmetric matrices and compare their eigenvalue histogram with the theoretical semicircle density.

### Circular Law

For non-Hermitian matrices with i.i.d. entries (Ginibre-type matrices), the eigenvalues of

$$
\frac{1}{\sqrt{n}}X
$$

converge to the **uniform distribution on the unit disk** in the complex plane.

Simulations show how the eigenvalue cloud approaches the unit disk as $n$ increases.

### Marchenko–Pastur Law

For rectangular random matrices \( X \in \mathbb{R}^{n \times p} \) with i.i.d. entries of mean \(0\) and variance \(1\), the eigenvalues of the sample covariance matrix

$$
\frac{1}{p} XX^{\top}
$$

converge, as \( n,p \to \infty \) with \( n/p \to \gamma \), to the **Marchenko–Pastur distribution** with density

$$
\rho_{\gamma}(x) = \frac{1}{2\pi \gamma x}\sqrt{(b-x)(x-a)}, \qquad x \in [a,b]
$$

where

$$
a = (1-\sqrt{\gamma})^2, \qquad b = (1+\sqrt{\gamma})^2.
$$

Simulations illustrate how the empirical spectrum of large covariance matrices approaches this limiting distribution.

### Spectral Radius Fluctuations

The spectral radius of the complex Ginibre ensemble exhibits **extreme value fluctuations**.

Let $r_n$ denote the spectral radius of

$$
\frac{1}{\sqrt{n}}G_n
$$

Then after appropriate centering and scaling, the fluctuations converge to a **Gumbel distribution** (Rider's theorem).

The repository includes simulations comparing empirical fluctuations with the theoretical Gumbel law.


## Installation

Clone the repository:

```bash
git clone https://github.com/ArnaudoB/Random-Matrix-Theory-simulations.git
cd Random-Matrix-Theory-simulations
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the simulations:

```bash
python circular_law.py
python wigner_semicircle.py
python spectral_radius_fluctuations.py
```
