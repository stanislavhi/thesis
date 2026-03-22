import numpy as np

def calculate_alpha(k_escape, delta_E, p, C_V, sigma):
    """
    Derives the coupling constant alpha from first principles.
    
    alpha = k_escape * delta_E * (ln2)^2 * (1-2p) / (C_V * sigma)
    """
    # Avoid division by zero
    if sigma < 1e-9: sigma = 1e-9
    if C_V < 1e-9: C_V = 1e-9
    
    ln2 = np.log(2)
    # Use absolute value for (1-2p) since coupling magnitude must be positive
    numerator = k_escape * delta_E * (ln2**2) * abs(1 - 2*p)
    denominator = C_V * sigma
    
    return numerator / denominator
