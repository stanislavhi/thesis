# Qwen-Thermodynamic Examples

This directory contains example scripts demonstrating various aspects of the Qwen-Thermodynamic framework.

## Quick Start

```bash
cd examples
python quick_start.py
```

This will run a complete demo including:
- Model initialization and training
- Inference with different temperatures
- Temperature comparison experiments

## Examples Overview

### 1. `quick_start.py`
A comprehensive example showing:
- Creating a thermodynamically-enhanced model
- Training with entropy regularization
- Generating text with controlled temperature
- Monitoring efficiency metrics

**Usage:**
```bash
python quick_start.py
```

**What it does:**
1. Initializes a small model (5000 vocab, 256 hidden dim) for fast testing
2. Trains for 5 epochs with entropy regularization
3. Demonstrates inference at different temperatures
4. Compares output quality across temperature settings

### 2. `train_visualization.py`
Monitors and visualizes training metrics including:
- Loss curves over epochs
- Entropy production rate
- Temperature evolution
- Efficiency metrics
- Heat distribution across layers
- Chaos injection frequency

**Usage:**
```bash
python train_visualization.py --metrics-path path/to/metrics.json
```

**Output files:**
- `training_metrics.png` - Multi-panel visualization of training progress
- `entropy_distribution.png` - Detailed entropy analysis
- `training_metrics.csv` - Raw metrics data for further analysis

## Model Architecture

The Qwen-Thermodynamic model extends standard transformers with:

### Key Components

1. **Entropy-Regularized Attention**
   ```python
   class EntropyRegularizedAttention(nn.Module):
       def __init__(self, entropy_weight=0.01):
           self.entropy_weight = entropy_weight
       
       def forward(self, q, k, v, mask=None):
           # Standard attention computation
           # Apply entropy regularization to output distribution
   ```

2. **Thermodynamic Monitor**
   Tracks system state including:
   - Temperature scale
   - Entropy production rate
   - Heat flow across layers
   - Efficiency metrics

3. **Chaos Injection**
   Introduces controlled randomness based on schedule:
   - `none`: No chaos injection
   - `linear`: Linear increase over training
   - `cosine`: Cosine-scheduled chaos
   - `adaptive`: Chaos based on system state

## Training Configuration

### Basic Training Setup

```python
from qwen.models.qwen_thermodynamic import QwenThermodynamicModel, QwenThermodynamicTrainer

# Initialize model
model = QwenThermodynamicModel(
    vocab_size=32000,      # Standard vocabulary size
    hidden_dim=512,        # Hidden dimension
    num_heads=8,           # Attention heads
    num_layers=6,          # Number of layers
    max_seq_len=256        # Maximum sequence length
)

# Initialize trainer with entropy regularization
trainer = QwenThermodynamicTrainer(
    model=model,
    lr=1e-4,              # Learning rate
    entropy_weight=0.01,   # Entropy regularization strength
    chaos_schedule="linear",  # Chaos injection schedule
    max_epochs=20          # Training epochs
)

# Train
for epoch in range(trainer.args.max_epochs):
    loss_dict = trainer.train_step()
```

### Inference Configuration

```python
from qwen.inference.qwen_thermodynamic_inferencer import QwenThermodynamicInferencer, InferenceConfig

config = InferenceConfig(
    max_length=512,       # Maximum output length
    temperature=1.0,      # Sampling temperature
    top_k=50,             # Top-k sampling
    use_chaos_injection=False  # Disable chaos for inference
)

inferencer = QwenThermodynamicInferencer(model, config)

# Generate text
input_ids = ...
output, diagnostics = inferencer.generate(input_ids)
```

## Understanding Entropy Regularization

Entropy regularization encourages diverse outputs by penalizing low-entropy (deterministic) probability distributions:

### Mathematical Formulation

The total loss becomes:

$$\mathcal{L}_{total} = \mathcal{L}_{cross\_entropy} + \lambda_{entropy} \cdot H(p)$$

Where:
- $\mathcal{L}_{cross\_entropy}$ is the standard cross-entropy loss
- $H(p)$ is the entropy of the output distribution
- $\lambda_{entropy}$ controls the regularization strength

### Effect on Training

- **High $\lambda_{entropy}$**: More diverse outputs, potentially lower coherence
- **Low $\lambda_{entropy}$**: More deterministic, higher quality but less creative
- **Optimal value**: Typically between 0.01 and 0.1 depending on task

## Chaos Injection

Chaos injection adds controlled randomness to break symmetry and encourage exploration:

### Schedules

1. **Linear Schedule**
   - Chaos increases linearly over training epochs
   - Good for breaking plateaus in loss curves
   
2. **Cosine Schedule**
   - Chaos follows cosine curve
   - High chaos at start, low chaos near convergence
   
3. **Adaptive Schedule**
   - Chaos based on system state (temperature, entropy rate)
   - Automatically adjusts to maintain optimal conditions

### Implementation

```python
# Enable linear chaos injection
trainer = QwenThermodynamicTrainer(
    model=model,
    lr=1e-4,
    entropy_weight=0.01,
    chaos_schedule="linear"
)
```

## Performance Metrics

The framework tracks several thermodynamic metrics:

### Efficiency Score

$$\text{Efficiency} = \frac{\text{Useful Work}}{\text{Total Energy Input}}$$

Calculated as:
- Useful work: Reduction in prediction error
- Total energy: Entropy production + computation cost

### Temperature Scale

Represents the "temperature" of the system, analogous to physical temperature:
- Low temperature (< 0.5): Deterministic, focused outputs
- Medium temperature (1.0): Balanced exploration/exploitation
- High temperature (> 2.0): Highly diverse, potentially incoherent outputs

### Entropy Production Rate

Measures how quickly the system produces entropy:
- Higher rates indicate more diversity but less efficiency
- Optimal rate depends on task requirements

## Visualization Tips

### Training Metrics

```bash
python train_visualization.py --metrics-path training_metrics.json
```

This generates:
1. **Loss Curve**: Shows convergence progress
2. **Entropy Rate**: Tracks diversity over time
3. **Temperature Evolution**: Monitors system temperature
4. **Efficiency Score**: Measures thermodynamic efficiency
5. **Heat Distribution**: Visualizes heat flow across layers
6. **Chaos Events**: Counts chaos injection events

### Customizing Plots

Edit `train_visualization.py` to customize:
- Plot colors and styles
- Figure size and layout
- Metrics to include/exclude
- Output resolution (dpi)

## Testing

Run the test suite:

```bash
python -m pytest tests/test_qwen_thermodynamic.py -v
```

Test coverage includes:
- Model architecture validation
- Entropy regularization correctness
- Heat flow monitoring accuracy
- Chaos injection schedules
- Inference engine functionality
- Edge cases and error handling

## Best Practices

### 1. Start Small
Begin with small models (500 vocab, 64 hidden dim) to verify setup before scaling up.

### 2. Monitor Entropy
Keep track of entropy production rate:
- Too low (< 0.1): Model may be overfitting or too deterministic
- Too high (> 5.0): Model may be underfitting or chaotic

### 3. Tune Temperature
Experiment with different temperatures for inference:
```python
for temp in [0.7, 1.0, 1.3]:
    config = InferenceConfig(temperature=temp)
    inferencer = QwenThermodynamicInferencer(model, config)
    # Generate and evaluate output quality
```

### 4. Use Appropriate Chaos Schedule
- Early training: Higher chaos for exploration
- Later training: Lower chaos for convergence
- Consider adaptive schedules for automatic tuning

### 5. Save Metrics Regularly
Enable metric logging during training to enable post-analysis:
```python
# Add to your training loop
trainer.monitor.log_metrics(epoch, loss_dict)
```

## Troubleshooting

### High Loss with Low Entropy
**Problem**: Model is too deterministic and not learning well.
**Solution**: Increase `entropy_weight` or adjust temperature.

### Diverse but Incoherent Output
**Problem**: Too much entropy regularization.
**Solution**: Decrease `entropy_weight` to 0.005-0.01.

### Training Plateau
**Problem**: Loss stops improving.
**Solution**: Enable chaos injection with linear schedule.

### Memory Issues
**Problem**: Out of memory errors during training.
**Solutions**:
- Reduce `max_seq_len`
- Use gradient accumulation
- Enable mixed precision training

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

This project is part of the Qwen-Thermodynamic research initiative. See LICENSE file for details.
