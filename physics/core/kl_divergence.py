import numpy as np

def calculate_kl_divergence(q, p):
    """
    Calculates the Kullback-Leibler Divergence D_KL(Q || P).
    epsilon = q * log(q/p) + (1-q) * log((1-q)/(1-p))
    """
    epsilon = 1e-9
    q = np.clip(q, epsilon, 1 - epsilon)
    p = np.clip(p, epsilon, 1 - epsilon)

    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
