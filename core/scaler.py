import math

class CosmologicalScaler:
    """
    Calculates the 'Ink Color' and 'Bit Capacity' for different physical systems
    based on the Holographic Principle and Landauer's Limit.
    """
    def __init__(self):
        self.k_B = 1.380649e-23 # Boltzmann constant
        self.h = 6.62607015e-34 # Planck constant
        self.c = 2.99792458e8   # Speed of light
        self.G = 6.67430e-11    # Gravitational constant
        self.l_P = math.sqrt((self.h * self.G) / (2 * math.pi * self.c**3)) # Planck length (approx)

    def analyze_system(self, name, radius_m, temperature_K):
        # 1. Bekenstein Bound (Bits)
        area = 4 * math.pi * radius_m**2
        bits = area / (4 * self.l_P**2 * math.log(2))
        
        # 2. Landauer Energy (Joules per bit flip)
        energy_per_bit = self.k_B * temperature_K * math.log(2)
        
        # 3. Ink Color (Wavelength of emitted photon)
        wavelength = (self.h * self.c) / energy_per_bit
        
        print(f"\n--- SYSTEM ANALYSIS: {name} ---")
        print(f"   Radius: {radius_m:.2e} m")
        print(f"   Temp:   {temperature_K:.2e} K")
        print(f"   >> CAPACITY: {bits:.2e} bits (Holographic Limit)")
        print(f"   >> COST:     {energy_per_bit:.2e} J/bit (Landauer Limit)")
        print(f"   >> INK:      {wavelength:.2e} m (Photon Wavelength)")
        
        return bits, energy_per_bit, wavelength
