#!/usr/bin/env python3
"""
Quick Start Example for Qwen-Thermodynamic.

This example demonstrates:
1. Creating a thermodynamically-enhanced model
2. Training with entropy regularization
3. Generating text with controlled temperature
4. Monitoring efficiency metrics
"""


def train_example():
    """Example training session."""
    import torch
    from qwen.models.qwen_thermodynamic import QwenThermodynamicModel, QwenThermodynamicTrainer
    
    print("=" * 70)
    print("Qwen-Thermodynamic Quick Start - Training Example")
    print("=" * 70)
    
    # Initialize model with reasonable defaults for quick testing
    print("\n1. Initializing Model...")
    model = QwenThermodynamicModel(
        vocab_size=5000,      # Small vocabulary for demo
        hidden_dim=256,       # Small hidden dim for speed
        num_heads=4,          # Fewer heads
        num_layers=2,         # Shallow network
        max_seq_len=128       # Short sequences
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer with entropy regularization
    print("\n2. Initializing Trainer...")
    trainer = QwenThermodynamicTrainer(
        model=model,
        lr=5e-4,              # Higher LR for quick training
        entropy_weight=0.01,   # Light entropy penalty
        chaos_schedule="none"  # No chaos for stability in demo
    )
    
    print("   ✓ Trainer ready")
    
    # Simulate a mini training session
    print("\n3. Running Mini Training Session...")
    for epoch in range(5):  # Just 5 epochs for demo
        loss_dict = trainer.train_step()
        
        state = trainer.monitor.compute_state()
        print(f"\n   Epoch {epoch+1}/5:")
        print(f"     Loss: {loss_dict['loss']:.4f}")
        print(f"     Entropy Rate: {state.entropy_production_rate:.4f}")
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'vocab_size': 5000, 'hidden_dim': 256},
        'epochs': 5
    }, '/tmp/qwen_demo_model.pt')
    
    print("\n4. Model saved to /tmp/qwen_demo_model.pt")


def infer_example():
    """Example inference session."""
    import torch
    from qwen.inference.qwen_thermodynamic_inferencer import QwenThermodynamicInferencer, InferenceConfig
    
    print("=" * 70)
    print("Qwen-Thermodynamic Quick Start - Inference Example")
    print("=" * 70)
    
    # Load model (using demo model from training example)
    print("\n1. Loading Model...")
    checkpoint = torch.load('/tmp/qwen_demo_model.pt', map_location='cpu')
    
    model = QwenThermodynamicModel(
        vocab_size=checkpoint.get('vocab_size', 5000),
        hidden_dim=checkpoint.get('hidden_dim', 256)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("   ✓ Model loaded")
    
    # Configure inference with different temperatures
    print("\n2. Configuring Inference...")
    
    config = InferenceConfig(
        max_length=64,
        temperature=1.0,      # Standard temperature
        top_k=50,             # Top-k sampling
        use_chaos_injection=False  # No chaos for demo
    )
    
    print("   ✓ Configuration ready")
    
    # Create inferencer
    inferencer = QwenThermodynamicInferencer(model, config)
    device = next(model.parameters()).device
    
    # Generate some text (using dummy tokens for demo)
    print("\n3. Generating Sample Output...")
    
    # Simple input sequence
    input_ids = torch.randint(100, 500, (1, 16)).to(device)
    
    with torch.no_grad():
        output, diagnostics = inferencer.generate(input_ids)
        
        print(f"\n   Output length: {len(output[0])} tokens")
        print(f"   Final Temperature: {diagnostics.get('final_temperature', 1.0):.4f}")
        print(f"   Avg Entropy: {diagnostics.get('avg_entropy', 0):.4f}")


def compare_temperatures():
    """Compare different temperature settings."""
    import torch
    from qwen.inference.qwen_thermodynamic_inferencer import QwenThermodynamicInferencer, InferenceConfig
    
    print("=" * 70)
    print("Temperature Comparison Example")
    print("=" * 70)
    
    # Load model
    checkpoint = torch.load('/tmp/qwen_demo_model.pt', map_location='cpu')
    model = QwenThermodynamicModel(
        vocab_size=checkpoint.get('vocab_size', 5000),
        hidden_dim=checkpoint.get('hidden_dim', 256)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test different temperatures
    temps = [0.7, 1.0, 1.3]
    
    for temp in temps:
        config = InferenceConfig(temperature=temp, max_length=64)
        inferencer = QwenThermodynamicInferencer(model, config)
        
        input_ids = torch.randint(100, 500, (1, 8)).to('cpu')
        
        with torch.no_grad():
            output, diagnostics = inferencer.generate(input_ids)
        
        print(f"\nTemperature {temp:.2f}:")
        print(f"   Avg Entropy: {diagnostics.get('avg_entropy', 0):.4f}")


if __name__ == "__main__":
    import torch
    
    print("\n" + "=" * 70)
    print("Qwen-Thermodynamic Quick Start Guide")
    print("=" * 70)
    
    # Run examples sequentially
    train_example()
    infer_example()
    compare_temperatures()
    
    print("\n" + "=" * 70)
    print("Quick Start Complete!")
    print("=" * 70)
