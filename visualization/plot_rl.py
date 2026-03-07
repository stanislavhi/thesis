import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def plot_rl_evolution():
    """
    Visualizes the RL agent's score, architectural evolution, and entropy production.
    """
    log_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/rl_training_log.csv'))
    
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: Could not find {log_file}. Run experiments/run_rl.py first.")
        return

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- TOP PLOT: Performance & Topology ---
    color = 'tab:green'
    ax1.set_ylabel('Score (Total Reward)', color=color)
    ax1.plot(df['episode'], df['avg_score'], color=color, linewidth=2, label='Avg Score (50 eps)')
    ax1.scatter(df['episode'], df['score'], color=color, alpha=0.1, s=10, label='Episode Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    
    ax1.axhline(y=195, color='green', linestyle=':', linewidth=1, label='Solved Threshold')

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Hidden Layer Size (Neurons)', color=color)  
    ax2.step(df['episode'], df['hidden_size'], color=color, where='post', linewidth=2, alpha=0.6, label='Brain Size')
    ax2.tick_params(axis='y', labelcolor=color)

    # Detect Mutations
    df['size_change'] = df['hidden_size'].diff().abs()
    mutations = df[df['size_change'] > 0]
    for episode in mutations['episode']:
        ax1.axvline(x=episode, color='tab:red', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_title('Thermodynamic RL: Evolution of Intelligence')

    # --- BOTTOM PLOT: Entropy Production (Heat) ---
    color = 'tab:red'
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Entropy Production (Grad Norm)', color=color)
    ax3.plot(df['episode'], df['entropy_production'], color=color, linewidth=1, alpha=0.8, label='Heat (Learning Rate)')
    ax3.fill_between(df['episode'], 0, df['entropy_production'], color=color, alpha=0.1)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid(True, which='both', linestyle='--', alpha=0.3)
    ax3.set_yscale('log')
    
    ax3.set_title('Thermodynamic Signature: Heat Dissipation during Learning')

    plt.tight_layout()
    
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/rl_evolution.png'))
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_rl_evolution()
