from collections import deque
import numpy as np

class ArchitectureMonitor:
    def __init__(self, window_size=20):
        self.loss_history = deque(maxlen=window_size)
    
    def update(self, loss):
        self.loss_history.append(loss.item())
        
    def is_plateaued(self, threshold_std=0.001):
        if len(self.loss_history) < self.loss_history.maxlen: return False
        loss_std = np.std(self.loss_history)
        return loss_std < threshold_std
