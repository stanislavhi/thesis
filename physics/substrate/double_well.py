class DoubleWellPotential:
    """
    V(x) = ax^4 - bx^2
    """
    def __init__(self, a=1.0, b=2.0):
        self.a = a
        self.b = b
        
    def energy(self, x):
        return self.a * x**4 - self.b * x**2
    
    def barrier_height(self):
        # Minima at x = +/- sqrt(b/2a)
        # Maxima at x = 0
        # E_min = -b^2 / 4a
        # E_max = 0
        # Delta E = b^2 / 4a
        return (self.b**2) / (4 * self.a)
