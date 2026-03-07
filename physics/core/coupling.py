import numpy as np

def calculate_alpha(k_escape, delta_E, p, C_V, sigma):
    """
    Derives the coupling constant alpha from first principles.
    
    alpha = k_escape * delta_E * (ln2)^2 * (1-2p) / (C_V * sigma)
    
    Note: Since alpha depends on sigma, and sigma depends on the dynamics (which depend on alpha),
    this is a self-consistency problem. For the prototype, we will use an initial estimate
    or an iterative solver.
    """
    # Avoid division by zero
    if sigma < 1e-9: sigma = 1e-9
    if C_V < 1e-9: C_V = 1e-9
    
    ln2 = np.log(2)
    numerator = k_escape * delta_E * (ln2**2) * (1 - 2*p)
    denominator = C_V * sigma
    
    return numerator / denominator
