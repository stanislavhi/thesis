import math

class LorenzGenerator:
    def __init__(self):
        self.x, self.y, self.z = 1.0, 1.0, 1.0
        self.sigma, self.rho, self.beta = 10.0, 28.0, 2.666
    
    def step(self):
        dt = 0.01
        dx = self.sigma * (self.y - self.x) * dt
        dy = (self.x * (self.rho - self.z) - self.y) * dt
        dz = (self.x * self.y - self.beta * self.z) * dt
        
        self.x += dx
        self.y += dy
        self.z += dz
        
        if abs(self.x) > 1e4 or abs(self.y) > 1e4 or abs(self.z) > 1e4 or math.isnan(self.x):
            self.x, self.y, self.z = 1.0, 1.0, 1.0 
            
        return self.z 

    def get_perturbation(self):
        chaos_val = (self.step() - 25.0) / 10.0 
        if math.isinf(chaos_val) or math.isnan(chaos_val): return 0.0
        return max(-5.0, min(5.0, chaos_val))
