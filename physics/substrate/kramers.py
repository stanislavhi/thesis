import numpy as np

def calculate_kramers_rate(barrier_height, temperature, attempt_freq=1.0):
    """
    k_escape = A * exp( -DeltaE / (k_B * T) )
    
    Args:
        barrier_height: Energy barrier ΔE between states
        temperature: Heat bath temperature T
        attempt_freq: Prefactor A = ω₀/(2π), the attempt frequency.
                      Default 1.0 for relative comparisons (Boltzmann factor only).
    """
    k_B = 1.0 # Normalized units
    if temperature < 1e-9: return 0.0
    
    return attempt_freq * np.exp(-barrier_height / (k_B * temperature))
