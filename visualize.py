import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_thermodynamics(log_file='thermodynamic_training_log.csv'):
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: Could not find {log_file}. Run test.py first.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Loss (The Energy Landscape)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)', color=color)
    ax1.plot(df['epoch'], df['loss'], color=color, linewidth=1.5, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log') # Log scale to see small improvements
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)

    # Plot Architecture Size (The Topology)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Hidden Layer Size (Neurons)', color=color)  
    ax2.step(df['epoch'], df['hidden_size'], color=color, where='post', linewidth=2, alpha=0.6, label='Network Size')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and Layout
    plt.title('Thermodynamic AI: Evolution of Structure vs. Performance')
    fig.tight_layout()  
    
    output_file = 'thermodynamic_evolution.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    # plt.show() # Uncomment if running locally with a display

if __name__ == "__main__":
    plot_thermodynamics()
