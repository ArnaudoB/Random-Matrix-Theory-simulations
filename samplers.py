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