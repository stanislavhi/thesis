import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import math
import os

# ==========================================
# 1. THE CHAOS ENGINE (Deterministic Chaos)
# ==========================================
class LorenzGenerator:
    """
    Generates a deterministic chaotic signal to drive topology changes.
    """
    def __init__(self):
        self.x, self.y, self.z = 1.0, 1.0, 1.0
        self.sigma, self.rho, self.beta = 10.0, 28.0, 2.666
    
    def step(self):
        dt = 0.01
        dx = self.sigma * (self.y - self.x) * dt
        dy = (self.x * (self.rho - self.z) - self.y) * dt
        dz = (self.x * self.z - self.beta * self.z) * dt
        
        self.x += dx
        self.y += dy
        self.z += dz
        
        # Safety Clamp to prevent OverflowError
        if abs(self.x) > 1e4 or abs(self.y) > 1e4 or abs(self.z) > 1e4 or math.isnan(self.x):
            self.x, self.y, self.z = 1.0, 1.0, 1.0 # Reset to initial state
            
        return self.z 

    def get_perturbation(self):
        """Maps the chaotic state to a perturbation factor."""
        # Normalize roughly around 0
        chaos_val = (self.step() - 25.0) / 10.0 
        
        # Safety Clamp
        if math.isinf(chaos_val) or math.isnan(chaos_val):
            return 0.0
        return max(-5.0, min(5.0, chaos_val)) # Clamp to reasonable range

# ==========================================
# 2. THE MONITOR (Thermostat)
# ==========================================
class ArchitectureMonitor:
    def __init__(self, window_size=20):
        self.loss_history = deque(maxlen=window_size)
    
    def update(self, loss):
        self.loss_history.append(loss.item())
        
    def is_plateaued(self, threshold_std=0.001):
        if len(self.loss_history) < self.loss_history.maxlen: return False
        
        # If variance of loss is very low, we are stuck in a local minimum
        loss_std = np.std(self.loss_history)
        return loss_std < threshold_std

# ==========================================
# 3. THE MUTATOR (Chaos Injector)
# ==========================================
class ChaosInjector:
    def __init__(self, generator):
        self.gen = generator
    
    def mutate_model(self, model):
        """
        The 'Chaotic Push': Structurally alters the network topology.
        Now includes 'Chemical Phase Shifts' (Activation Function Mutation).
        """
        if not isinstance(model, nn.Sequential):
             return model
             
        first_layer = model[0]
        last_layer = model[-1]
        
        input_dim = first_layer.in_features
        current_hidden = first_layer.out_features
        output_dim = last_layer.out_features
        
        # Get chaos factor
        chaos = self.gen.get_perturbation()
        
        # Determine new hidden size (Topological Mutation)
        try:
            change = int(chaos * 10)
        except OverflowError:
            change = 0

        if change == 0: change = 1 if np.random.random() > 0.5 else -1
        
        new_hidden = current_hidden + change
        new_hidden = max(16, min(new_hidden, 256)) 
        
        # Chemical Phase Shift (Activation Function Mutation)
        # If chaos is very high (> 1.5), swap the activation function
        current_act = model[1]
        new_act = current_act
        act_name = "Unchanged"
        
        if abs(chaos) > 1.2:
            if isinstance(current_act, nn.ReLU):
                new_act = nn.Tanh()
                act_name = "ReLU -> Tanh"
            elif isinstance(current_act, nn.Tanh):
                new_act = nn.GELU()
                act_name = "Tanh -> GELU"
            else:
                new_act = nn.ReLU()
                act_name = "GELU -> ReLU"
        
        print(f"   >>> CHAOS INJECTION: Resizing {current_hidden}->{new_hidden} | Phase Shift: {act_name} (Chaos: {chaos:.3f})", flush=True)
        
        # Create new architecture (The Leap)
        new_model = nn.Sequential(
            nn.Linear(input_dim, new_hidden),
            new_act,
            nn.Linear(new_hidden, output_dim)
        )
        
        # Transfer Knowledge (The "Soft" Leap)
        self._transfer_weights(model, new_model)
        
        return new_model

    def _transfer_weights(self, old_model, new_model):
        """
        Preserves learned features by copying overlapping weights.
        """
        try:
            old_l1 = old_model[0]
            new_l1 = new_model[0]
            
            n_in = min(old_l1.in_features, new_l1.in_features)
            n_out = min(old_l1.out_features, new_l1.out_features)
            
            with torch.no_grad():
                new_l1.weight[:n_out, :n_in] = old_l1.weight[:n_out, :n_in]
                new_l1.bias[:n_out] = old_l1.bias[:n_out]
                
            old_l2 = old_model[2]
            new_l2 = new_model[2]
            
            n_in_2 = min(old_l2.in_features, new_l2.in_features)
            n_out_2 = min(old_l2.out_features, new_l2.out_features)
            
            with torch.no_grad():
                new_l2.weight[:n_out_2, :n_in_2] = old_l2.weight[:n_out_2, :n_in_2]
                new_l2.bias[:n_out_2] = old_l2.bias[:n_out_2]
                
            print(f"   >>> MEMORY PRESERVED: Transferred weights for {n_out} neurons.", flush=True)
        except Exception as e:
            print(f"   >>> MEMORY LOSS: Could not transfer weights ({e})", flush=True)

# ==========================================
# 4. DATA GENERATOR (Mackey-Glass)
# ==========================================
def generate_mackey_glass(n_samples=1200, tau=17):
    beta = 0.2
    gamma = 0.1
    n = 10
    delta_t = 0.1
    
    history_len = int(tau / delta_t)
    timeseries = [1.2] * (history_len + n_samples)
    
    for i in range(history_len, len(timeseries) - 1):
        x_t = timeseries[i]
        x_tau = timeseries[i - history_len]
        
        dx = (beta * x_tau / (1 + x_tau**n) - gamma * x_t) * delta_t
        timeseries[i+1] = x_t + dx
        
    data = np.array(timeseries[history_len:])
    
    window_size = 10
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).unsqueeze(1)

# ==========================================
# 5. MAIN LOOP
# ==========================================

def train_chaotic_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}", flush=True)
    
    chaos_gen = LorenzGenerator()
    injector = ChaosInjector(chaos_gen)
    monitor = ArchitectureMonitor(window_size=15)
    
    print("Generating Mackey-Glass Chaotic Time Series...", flush=True)
    X_data, y_data = generate_mackey_glass()
    X_data, y_data = X_data.to(device), y_data.to(device)
    
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    epochs = 1500
    
    best_loss = float('inf')
    best_arch = 0
    loss_log = []
    
    print("Starting Thermodynamic Training Loop...", flush=True)
    
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()
        outputs = model(X_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        monitor.update(loss)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_arch = model[0].out_features
            
        loss_log.append(f"{epoch},{current_loss},{model[0].out_features}")
        
        if epoch > 10 and monitor.is_plateaued(threshold_std=0.0001):
            print(f"\n[Epoch {epoch}] Thermodynamic Barrier Detected (Loss Stagnation).", flush=True)
            
            model = injector.mutate_model(model).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            monitor.loss_history.clear()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {current_loss:.6f} | Chaos Z: {chaos_gen.step():.2f}", flush=True)

    with open("thermodynamic_training_log.csv", "w") as f:
        f.write("epoch,loss,hidden_size\n")
        f.write("\n".join(loss_log))
        
    print(f"\nTraining Complete.", flush=True)
    print(f"Best Loss Achieved: {best_loss:.6f} with Hidden Size: {best_arch}", flush=True)
    print("Detailed log saved to 'thermodynamic_training_log.csv'", flush=True)

if __name__ == "__main__":
    train_chaotic_agent()
