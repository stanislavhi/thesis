import numpy as np

def calculate_kramers_rate(barrier_height, temperature):
    """
    k_escape = A * exp( -DeltaE / (k_B * T) )
    """
    k_B = 1.0 # Normalized units
    if temperature < 1e-9: return 0.0
    
    return np.exp(-barrier_height / (k_B * temperature))
