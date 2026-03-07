import numpy as np

def calculate_schottky_heat_capacity(delta_E, temperature):
    """
    C_V for a two-level system (Schottky Anomaly).
    C_V = k_B * (DeltaE / k_B T)^2 * exp(DeltaE / k_B T) / (1 + exp(DeltaE / k_B T))^2
    """
    k_B = 1.0 # Normalized units
    if temperature < 1e-9: return 0.0
    
    x = delta_E / (k_B * temperature)
    
    # Avoid overflow for large x
    if x > 100: return 0.0
    
    exp_x = np.exp(x)
    return k_B * (x**2) * exp_x / ((1 + exp_x)**2)
