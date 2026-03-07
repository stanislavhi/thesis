import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def plot_swarm_evolution():
    """
    Visualizes the training loss of the Thermodynamic Swarm.
    """
    log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/swarm_log.csv'))
    
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: Could not find {log_file}. Run experiments/run_swarm.py first.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Loss (The Energy Landscape)
    color = 'tab:purple'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Swarm Loss (MSE)', color=color)
    ax1.plot(df['epoch'], df['loss'], color=color, linewidth=1.5, label='Swarm Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log') # Log scale to see small improvements
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)

    # Find mutation points by detecting large jumps in loss
    df['loss_jump'] = df['loss'].diff().abs()
    mutation_threshold = df['loss_jump'].mean() * 10 # Heuristic for a large jump
    mutation_points = df[df['loss_jump'] > mutation_threshold]

    for epoch in mutation_points['epoch']:
        ax1.axvline(x=epoch, color='tab:red', linestyle='--', linewidth=1, alpha=0.7, label='Chaos Injection')

    # Clean up duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    # Title and Layout
    plt.title('Thermodynamic Swarm: Collective Loss vs. Chaotic Mutations')
    fig.tight_layout()  
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/swarm_evolution.png'))
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_swarm_evolution()
