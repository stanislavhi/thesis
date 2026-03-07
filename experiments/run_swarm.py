import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.chaos import LorenzGenerator
from core.monitor import ArchitectureMonitor
from core.scaler import CosmologicalScaler
from agents.swarm import SwarmAgent, HolographicChannel, HolographicAggregator, ChaosInjector

def generate_mackey_glass(n_samples=1200):
    beta, gamma, n, tau, delta_t = 0.2, 0.1, 10, 17, 0.1
    history_len = int(tau / delta_t)
    timeseries = [1.2] * (history_len + n_samples)
    for i in range(history_len, len(timeseries) - 1):
        x_t = timeseries[i]
        x_tau = timeseries[i - history_len]
        dx = (beta * x_tau / (1 + x_tau**n) - gamma * x_t) * delta_t
        timeseries[i+1] = x_t + dx
    data = np.array(timeseries[history_len:])
    X, y = [], []
    window_size = 10
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).unsqueeze(1)

def train_swarm():
    # 0. Run Cosmological Analysis
    scaler = CosmologicalScaler()
    uni_bits, _, _ = scaler.analyze_system("Observable Universe", 4.4e26, 2.725)
    bh_bits, _, _ = scaler.analyze_system("Solar Mass Black Hole", 2953, 6e-8)
    brain_bits, _, _ = scaler.analyze_system("Human Brain", 0.07, 310)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing Thermodynamic Swarm on {device}...", flush=True)
    
    # Initialize Channel with Brain-like capacity
    channel = HolographicChannel(brain_bits).to(device)
    thought_size = channel.thought_vector_size
    
    chaos_gen = LorenzGenerator()
    injector = ChaosInjector(chaos_gen)
    monitor = ArchitectureMonitor(window_size=20)
    
    # Data
    X_data, y_data = generate_mackey_glass()
    X_data, y_data = X_data.to(device), y_data.to(device)
    
    # Initialize Swarm (3 Agents)
    agents = [
        SwarmAgent(4, 16, thought_size, "A").to(device),
        SwarmAgent(3, 16, thought_size, "B").to(device),
        SwarmAgent(4, 16, thought_size, "C").to(device)
    ]
    
    # Aggregator takes 3 * thought_size inputs
    aggregator = HolographicAggregator(len(agents) * thought_size, 32).to(device)
    
    params = list(aggregator.parameters()) + list(channel.parameters())
    for a in agents: params += list(a.parameters())
    optimizer = optim.Adam(params, lr=0.01)
    criterion = nn.MSELoss()
    
    epochs = 100000
    loss_log = []
    
    print("Starting Swarm Evolution...", flush=True)
    
    for epoch in range(epochs):
        # Forward Pass
        x1 = X_data[:, 0:4]
        x2 = X_data[:, 3:6]
        x3 = X_data[:, 6:10]
        
        thoughts = [
            agents[0](x1),
            agents[1](x2),
            agents[2](x3)
        ]
        
        # Pass thoughts through the noisy, constrained channel
        noisy_thoughts = channel(thoughts, monitor.loss_history[-1] if monitor.loss_history else 0.1)
        
        # Aggregator Prediction
        prediction = aggregator(noisy_thoughts)
        
        loss = criterion(prediction, y_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        monitor.update(loss)
        loss_log.append(f"{epoch},{loss.item()}")
        
        # Thermodynamic Check
        if epoch > 20 and monitor.is_plateaued(threshold_std=0.00005):
            print(f"\n[Epoch {epoch}] Swarm Stagnation Detected.", flush=True)
            
            agents, aggregator = injector.mutate_swarm(agents, aggregator)
            
            # Re-init optimizer
            for a in agents: a.to(device)
            aggregator.to(device)
            params = list(aggregator.parameters()) + list(channel.parameters())
            for a in agents: params += list(a.parameters())
            optimizer = optim.Adam(params, lr=0.01)
            monitor.loss_history.clear()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Swarm Loss: {loss.item():.6f} | Chaos: {chaos_gen.step():.2f}", flush=True)

    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/swarm_log.csv'))
    with open(log_path, "w") as f:
        f.write("epoch,loss\n")
        f.write("\n".join(loss_log))
    print(f"Swarm training complete. Log saved to {log_path}", flush=True)

if __name__ == "__main__":
    train_swarm()
