#!/usr/bin/env python3
"""
Training Visualization Script.

Monitors and visualizes thermodynamic metrics during training:
- Loss curves
- Entropy production rate
- Temperature evolution
- Efficiency metrics
- Heat distribution across layers
"""


def plot_training_metrics(metrics_history, output_path='metrics.png'):
    """Create visualization of training metrics."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Loss over epochs
    ax = axes[0, 0]
    if 'loss' in metrics_history:
        ax.plot(metrics_history['loss'], label='Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Entropy production rate
    ax = axes[0, 1]
    if 'entropy_rate' in metrics_history:
        ax.plot(metrics_history['entropy_rate'], label='Entropy Rate', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Bits/token')
        ax.set_title('Entropy Production Rate')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Temperature evolution
    ax = axes[0, 2]
    if 'temperature' in metrics_history:
        ax.plot(metrics_history['temperature'], label='Temperature', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Temperature')
        ax.set_title('Sampling Temperature')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency metrics
    ax = axes[1, 0]
    if 'efficiency' in metrics_history:
        ax.plot(metrics_history['efficiency'], label='Efficiency', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Efficiency Score')
        ax.set_title('Thermodynamic Efficiency')
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Heat distribution (simplified)
    ax = axes[1, 1]
    if 'heat_by_layer' in metrics_history and len(metrics_history['heat_by_layer']) > 0:
        for i, heat_list in enumerate(metrics_history['heat_by_layer']):
            ax.plot(range(len(heat_list)), heat_list, label=f'Layer {i}', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Heat Units')
        ax.set_title('Heat Distribution by Layer')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Chaos injection events
    ax = axes[1, 2]
    if 'chaos_events' in metrics_history and len(metrics_history['chaos_events']) > 0:
        epochs = list(range(1, len(metrics_history['loss']) + 1))
        chaos_counts = [metrics_history['chaos_events'][i] for i in range(len(metrics_history['chaos_events']))]
        ax.bar(epochs, chaos_counts, color='orange', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Chaos Events')
        ax.set_title('Chaos Injection Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")


def plot_entropy_distribution(entropy_by_token):
    """Plot entropy distribution across tokens."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Histogram of entropy values
    ax = axes[0, 0]
    if len(entropy_by_token) > 0:
        ax.hist(entropy_by_token, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Entropy (bits/token)')
        ax.set_ylabel('Frequency')
        ax.set_title('Entropy Distribution Histogram')
    
    # Box plot by token position
    ax = axes[0, 1]
    if len(entropy_by_token) > 0:
        positions = list(range(len(entropy_by_token)))
        bp = ax.boxplot(entropy_by_token, labels=positions[:20], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Entropy (bits/token)')
        ax.set_title('Entropy by Token Position')
    
    # Scatter plot of entropy vs step
    ax = axes[1, 0]
    if len(entropy_by_token) > 0:
        steps = list(range(len(entropy_by_token)))
        scatter = ax.scatter(steps, entropy_by_token[:20], alpha=0.5)
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Entropy (bits/token)')
        ax.set_title('Entropy vs Token Position')
    
    # Line plot of rolling average
    ax = axes[1, 1]
    if len(entropy_by_token) > 0:
        window_size = min(20, len(entropy_by_token))
        rolling_avg = []
        for i in range(window_size - 1, len(entropy_by_token)):
            rolling_avg.append(sum(entropy_by_token[i-window_size+1:i+1]) / window_size)
        
        ax.plot(range(len(rolling_avg)), rolling_avg, linewidth=2)
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Avg Entropy (bits/token)')
        ax.set_title('Rolling Average Entropy')
    
    plt.tight_layout()
    plt.savefig('entropy_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved entropy distribution to entropy_distribution.png")


def monitor_training(trainer, log_interval=5):
    """Monitor training and collect metrics."""
    import time
    
    print("\n" + "=" * 70)
    print("Training Monitor Active")
    print("=" * 70)
    
    start_time = time.time()
    metrics_history = {
        'loss': [],
        'entropy_rate': [],
        'temperature': [],
        'efficiency': [],
        'chaos_events': []
    }
    
    for epoch in range(trainer.args.epochs):
        loss_dict = trainer.train_step()
        
        state = trainer.monitor.compute_state()
        
        # Collect metrics at intervals
        if (epoch + 1) % log_interval == 0:
            print(f"\nEpoch {epoch+1}/{trainer.args.epochs}:")
            print(f"  Loss: {loss_dict['loss']:.4f}")
            print(f"  Entropy Rate: {state.entropy_production_rate:.4f}")
            print(f"  Efficiency: {state.efficiency:.4f}")
            
            metrics_history['loss'].append(loss_dict['loss'])
            metrics_history['entropy_rate'].append(state.entropy_production_rate)
            metrics_history['temperature'].append(state.temperature_scale)
            metrics_history['efficiency'].append(state.efficiency)
            metrics_history['chaos_events'].append(len(trainer.monitor.chaos_events))
        
        # Update chaos events list
        trainer.monitor.chaos_events.clear()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_metrics(metrics_history, 'qwen_training_metrics.png')
    plot_entropy_distribution(trainer.monitor.entropy_by_token)
    
    print(f"\n✓ Training complete in {time.time() - start_time:.2f}s")


def generate_report(metrics_history):
    """Generate a text report of training metrics."""
    import pandas as pd
    
    if not metrics_history:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(metrics_history)
    
    print("\n" + "=" * 70)
    print("Training Report")
    print("=" * 70)
    
    print(f"\nTotal Epochs: {len(df)}")
    print(f"Final Loss: {df['loss'].iloc[-1]:.4f}")
    print(f"Avg Entropy Rate: {df['entropy_rate'].mean():.4f}")
    print(f"Max Efficiency: {df['efficiency'].max():.4f}")
    
    # Save to CSV if metrics_history has enough data
    if len(df) > 10:
        output_path = 'training_metrics.csv'
        df.to_csv(output_path, index=False)
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    # This script would be used during training
    # In practice, you'd call these functions from within the trainer
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Qwen-Thermodynamic Training Monitor')
    parser.add_argument('--metrics-path', type=str, default='training_metrics.json',
                        help='Path to metrics JSON file')
    
    args = parser.parse_args()
    
    # Load and visualize existing metrics
    import json
    
    with open(args.metrics_path) as f:
        metrics_history = json.load(f)
    
    plot_training_metrics(metrics_history, 'training_visualization.png')
    generate_report(metrics_history)
