#!/usr/bin/env python3
"""
Qwen-Thermodynamic - Main Entry Point.

A thermodynamically-enhanced transformer model that incorporates:
- Entropy-based regularization for coherent generation
- Heat flow monitoring to identify computational bottlenecks
- Chaos injection for exploration in uncertain regions
- Free energy optimization during training and inference

Usage:
    # Training
    python main.py train --epochs 100 --hidden-dim 1024
    
    # Inference
    python main.py infer --model models/best_model.pt
    
    # Evaluate
    python main.py eval --model models/best_model.pt
"""

import argparse
import sys
import torch


def load_package():
    """Load the Qwen-Thermodynamic package."""
    try:
        from qwen.models.qwen_thermodynamic import (
            QwenThermodynamicModel, 
            QwenThermodynamicTrainer
        )
        return True
    except ImportError as e:
        print(f"Warning: Could not load Qwen-Thermodynamic package: {e}")
        print("Installing from source...")
        
        # Try to install in development mode
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
            return True
        except Exception as e2:
            print(f"Installation failed: {e2}")
            print("Using fallback imports...")
            
            # Fallback for testing
            sys.path.insert(0, '/Users/stanislavhiznicenco/IdeaProjects/thesis')
            from thesis.qwen.models.qwen_thermodynamic import (
                QwenThermodynamicModel, 
                QwenThermodynamicTrainer
            )
            return True


def train(args):
    """Train the thermodynamically-enhanced model."""
    print("=" * 70)
    print("Qwen-Thermodynamic Training")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Initialize model
    print("\nInitializing Model...")
    model = QwenThermodynamicModel(
        vocab_size=5000,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    # Initialize trainer
    print("\nInitializing Trainer...")
    trainer = QwenThermodynamicTrainer(
        model=model,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        chaos_schedule="linear" if args.use_chaos else "none",
        device=device
    )
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    log_interval = 10
    
    for epoch in range(args.epochs):
        loss_dict = trainer.train_step()
        
        if (epoch + 1) % log_interval == 0:
            state = trainer.monitor.compute_state()
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Loss: {loss_dict['loss']:.4f}")
            print(f"  Entropy Rate: {state.entropy_production_rate:.4f}")
            print(f"  Efficiency: {state.efficiency:.4f}")
    
    # Save model
    save_path = f"models/qwen_thermodynamic_final_{args.hidden_dim}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'epochs': args.epochs
    }, save_path)
    
    print(f"\n✓ Model saved to {save_path}")


def infer(args):
    """Run inference with the trained model."""
    from qwen.inference.qwen_thermodynamic_inferencer import QwenThermodynamicInferencer
    
    # Load model
    checkpoint = torch.load(args.model, map_location='cpu')
    
    model = QwenThermodynamicModel(
        vocab_size=checkpoint.get('vocab_size', 5000),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup inferencer
    config = QwenThermodynamicInferencer.InferenceConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        max_length=args.max_length
    )
    
    inferencer = QwenThermodynamicInferencer(model, config)
    
    # Generate prompt (simple test for now)
    print("\nGenerating response...")
    
    # Create dummy input - in practice, this would be user input
    if args.prompt:
        input_ids = torch.tensor([[ord(c) % 5000 for c in args.prompt[:100]]]).to('cpu')
    else:
        input_ids = torch.randint(0, 5000, (1, 20)).to('cpu')
    
    with torch.no_grad():
        output, diagnostics = inferencer.generate(input_ids)
        
        # Decode and print
        text = ''.join([chr(int(token.item()) % 128 + 32) for token in output[-256:]])
        print("\n" + "=" * 70)
        print(text)
        print("=" * 70)
        
        # Print diagnostics
        print(f"\nDiagnostics:")
        print(f"  Final Temperature: {diagnostics.get('final_temperature', args.temperature):.4f}")
        print(f"  Avg Entropy: {diagnostics.get('avg_entropy', 0):.4f}")


def eval(args):
    """Evaluate the model."""
    from qwen.experiments.train_thermodynamic_qwen import evaluate_model
    
    # Parse args for evaluation
    import sys
    original_argv = sys.argv
    sys.argv = ['main.py', 'eval'] + [f"--model={args.model}"] + \
               ["--hidden-dim", str(args.hidden_dim), "--num-heads", str(args.num_heads)]
    
    evaluate_model(type('Args', (), {
        'model': args.model,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'seq_len': args.max_seq_len or 512
    }))


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Thermodynamic: Entropy-Regularized Transformer"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--hidden-dim', type=int, default=1024)
    train_parser.add_argument('--num-heads', type=int, default=8)
    train_parser.add_argument('--num-layers', type=int, default=4)
    train_parser.add_argument('--max-seq-len', type=int, default=512)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--entropy-weight', type=float, default=0.01)
    train_parser.add_argument('--use-chaos', action='store_true')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    infer_parser.add_argument('--prompt', type=str, default='', help='Input prompt')
    infer_parser.add_argument('--temperature', type=float, default=1.0)
    infer_parser.add_argument('--top-k', type=int, default=50)
    infer_parser.add_argument('--max-length', type=int, default=2048)
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--hidden-dim', type=int, default=1024)
    eval_parser.add_argument('--num-heads', type=int, default=8)
    eval_parser.add_argument('--max-seq-len', type=int, default=512)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load package first
    load_package()
    
    # Execute command
    if args.command == 'train':
        train(args)
    elif args.command == 'infer':
        infer(args)
    elif args.command == 'eval':
        eval(args)


if __name__ == "__main__":
    main()
