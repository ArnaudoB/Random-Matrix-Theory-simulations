import numpy as np

def semicircle_density(x):
    # support [-2,2]
    return (1/(2*np.pi)) * np.sqrt(np.maximum(0.0, 4 - x**2))

def circular_law_density(x, y):
    r2 = x**2 + y**2
    return (1/np.pi) * (r2 <= 1)

def gumbel_density(x):
        return np.exp(-(x + np.exp(-x)))

def marchenko_pastur_density(x, c):
    a = (1 - np.sqrt(c))**2
    b = (1 + np.sqrt(c))**2
    return (1 - 1/c) * (x == 0) * (c < 1) + (1/(2*np.pi*c*x)) * np.sqrt(np.maximum(0.0, (b-x)*(x-a)))