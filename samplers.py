import numpy as np


def sample_gue(n, rng):
    X = rng.normal(size=(n, n))
    Y = rng.normal(size=(n, n))
    Z = (X + 1j * Y) / np.sqrt(2) 
    return (Z + Z.conj().T)/np.sqrt(2)


def sample_covariance(n, p, rng):
    X = rng.normal(size=(p, n))
    Y = rng.normal(size=(p, n))
    Z = (X + 1j * Y) / np.sqrt(2)
    return Z @ Z.conj().T


def sample_ginibre(n, rng):
    X = rng.normal(size=(n, n))
    Y = rng.normal(size=(n, n))
    Z = (X + 1j * Y) / np.sqrt(2)
    return Z


def sample_bernoulli(n, rng):
    return rng.choice([-1, 1], size=(n, n))