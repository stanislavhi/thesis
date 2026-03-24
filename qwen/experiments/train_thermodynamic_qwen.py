#!/usr/bin/env python3
"""
Thermodynamically-Enhanced Qwen Training.

This script trains a Qwen model with thermodynamic learning mechanisms,
including entropy regularization and chaos-based curriculum learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import argparse
import os
from datetime import datetime


# Import from local modules (adjust paths if needed)
try:
    from qwen.models.qwen_thermodynamic import QwenThermodynamicModel, QwenThermodynamicTrainer
    from qwen.utils.thermodynamic_monitor import ThermodynamicMonitor
except ImportError:
    # Fallback for testing without installed package
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from qwen.models.qwen_thermodynamic import QwenThermodynamicModel, QwenThermodynamicTrainer
    from qwen.utils.thermodynamic_monitor import ThermodynamicMonitor


def create_dummy_dataset(batch_size: int = 4, seq_len: int = 128):
    """Create dummy dataset for testing."""
    vocab_size = 5000
    
    def collate_fn(data):
        return {
            'input_ids': torch.stack([torch.randint(0, vocab_size, (seq_len,)) for _ in range(batch_size)]),
            'labels': torch.stack([torch.randint(0, vocab_size, (seq_len,)) for _ in range(batch_size)])
        }
    
    return collate_fn


def train_model(args):
    """Main training loop."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print(f"\nInitializing Qwen-Thermodynamic Model:")
    print(f"  - Hidden dim: {args.hidden_dim}")
    print(f"  - Num heads: {args.num_heads}")
    print(f"  - Num layers: {args.num_layers}")
    
    model = QwenThermodynamicModel(
        vocab_size=5000,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    # Initialize trainer
    print(f"\nInitializing Trainer:")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Entropy weight: {args.entropy_weight}")
    
    trainer = QwenThermodynamicTrainer(
        model=model,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        chaos_schedule="linear" if args.use_chaos else "none"
    )
    
    # Setup monitoring
    monitor = ThermodynamicMonitor(window_size=100)
    
    # Create dataset
    collate_fn = create_dummy_dataset(args.batch_size, args.seq_len)
    dataloader = torch.utils.data.DataLoader(
        [None],  # Dummy dataset
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    best_loss = float('inf')
    log_interval = 10
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        # Use chaos injection every N epochs
        use_chaos = args.use_chaos and epoch % args.chaos_every_n == 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
                
            # Train step
            loss_dict = trainer.train_step(batch, chaos_inject=use_chaos)
            
            total_loss += loss_dict['loss']
            
            # Update monitor
            gradient_norm = torch.norm(model.output_proj.weight).item()
            monitor.update(loss_dict, gradient_norm)
        
        avg_loss = total_loss / args.batch_size
        
        # Compute thermodynamic state
        state = monitor.compute_state()
        
        # Log progress
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Entropy Rate: {state.entropy_production_rate:.4f}")
            print(f"  Temperature: {state.temperature_local:.4f}")
            print(f"  Efficiency: {state.efficiency:.4f}")
            if use_chaos:
                print(f"  ⚠️ Chaos injection active!")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = f"models/qwen_thermodynamic_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'epoch': epoch
            }, save_path)
            print(f"\n✓ Saved best model to {save_path}")
    
    # Final summary
    final_state = monitor.compute_state()
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Loss: {avg_loss:.4f}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Entropy Rate: {final_state.entropy_production_rate:.4f}")
    print(f"Final Efficiency: {final_state.efficiency:.4f}")
    
    # Save final model
    save_path = f"models/qwen_thermodynamic_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_loss': avg_loss,
        'epochs': args.epochs
    }, save_path)
    print(f"\n✓ Saved final model to {save_path}")


def evaluate_model(args):
    """Evaluate trained thermodynamic Qwen model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best model
    checkpoint_path = "models/qwen_thermodynamic_best_*.pt"
    import glob
    checkpoints = glob.glob(checkpoint_path)
    
    if not checkpoints:
        print("No trained models found. Run training first.")
        return
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"\nLoading model from {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model = QwenThermodynamicModel(
        vocab_size=5000,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = args.seq_len if hasattr(args, 'seq_len') else 512
    
    with torch.no_grad():
        input_ids = torch.randint(0, 5000, (batch_size, seq_len)).to(device)
        
        # Evaluate without chaos injection
        logits, diagnostics = model(input_ids, chaos_inject=False)
        
        print(f"\nEvaluation Results:")
        print(f"  Entropy Rate: {diagnostics['entropy']:.4f}")
        print(f"  Temperature Scale: {diagnostics['temperature_scale']:.4f}")
        
    # Save evaluation metrics
    eval_path = "logs/qwen_evaluation.json"
    import json
    with open(eval_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    print(f"\n✓ Saved evaluation metrics to {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="Train thermodynamically-enhanced Qwen model")
    
    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--entropy-weight", type=float, default=0.01, 
                       help="Entropy regularization weight")
    
    # Chaos injection
    parser.add_argument("--use-chaos", action="store_true", 
                       help="Enable chaos-based curriculum learning")
    parser.add_argument("--chaos-every-n", type=int, default=25,
                       help="Inject chaos every N epochs")
    
    # Other
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate existing model")
    
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_model(args)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
