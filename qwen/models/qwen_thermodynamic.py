"""
Qwen Thermodynamic Model Integration

This module implements thermodynamically-inspired variants of Qwen architecture,
integrating chaos theory and entropy-based learning mechanisms.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ThermodynamicAttention(nn.Module):
    """
    Attention mechanism with thermodynamic constraints.
    
    Incorporates:
    - Entropy regularization for diverse attention distributions
    - Chaos injection for exploration in high-uncertainty regions
    - Temperature scaling based on local gradient norms
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, entropy_weight: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Standard attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Thermodynamic parameters
        self.entropy_weight = entropy_weight
        self.temperature_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, chaos_inject: bool = False, z_value: float = 0.0) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with thermodynamic monitoring.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            chaos_inject: Whether to inject chaos into attention weights
            z_value: Chaos parameter from Lorenz attractor
            
        Returns:
            Attention output and diagnostics dict
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute attention scores with temperature scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) / self.temperature_scale
        
        # Thermodynamic entropy regularization
        probs = torch.softmax(scores, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        
        # Apply chaos injection if requested
        if chaos_inject and abs(z_value) > 0.5:
            noise_mask = self._generate_chaos_noise(batch_size, seq_len, z_value)
            probs = torch.where(
                noise_mask.abs() > 0.1, 
                probs * (1 + noise_mask), 
                probs
            )
        
        # Weighted attention with entropy consideration
        attention_output = torch.matmul(probs, v)
        
        diagnostics = {
            'entropy': entropy.mean().item(),
            'temperature': self.temperature_scale.item(),
            'chaos_applied': chaos_inject and abs(z_value) > 0.5
        }
        
        return attention_output, diagnostics
    
    def _generate_chaos_noise(self, batch_size: int, seq_len: int, z: float) -> torch.Tensor:
        """Generate noise based on Lorenz attractor Z value."""
        # Gaussian noise with variance controlled by |Z|
        scale = 1 + abs(z) * 2
        return torch.randn(batch_size, seq_len).mul(scale)


class ThermodynamicTransformerBlock(nn.Module):
    """
    Transformer block with thermodynamic learning mechanisms.
    
    Features:
    - Multi-head attention with entropy regularization
    - Feed-forward network with chaos-based gating
    - Residual connections with temperature scaling
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, ff_dim: Optional[int] = None):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = 4 * d_model
        
        self.attention = ThermodynamicAttention(d_model, num_heads)
        
        # Feed-forward with thermodynamic gating
        self.ffn1 = nn.Linear(d_model, ff_dim)
        self.ffn2 = nn.Linear(ff_dim, d_model)
        
        # Gating mechanism controlled by chaos parameter
        self.gate_proj = nn.Linear(d_model, 4)  # 4 gates: attention_scale, ffn_scale, residual_scale, entropy_scale
        
    def forward(self, x: torch.Tensor, diagnostics_history: list = None) -> Tuple[torch.Tensor, dict]:
        """Forward pass with thermodynamic adaptation."""
        if diagnostics_history is None:
            diagnostics_history = []
        
        # Compute gating parameters based on input statistics
        gate_input = x.mean(dim=1).unsqueeze(1)  # (batch, 1, d_model)
        gates = torch.sigmoid(self.gate_proj(gate_input))  # (batch, 4)
        
        attention_scale = gates[:, 0].unsqueeze(1).unsqueeze(2)
        ffn_scale = gates[:, 1].unsqueeze(1).unsqueeze(2)
        
        # Attention with chaos injection
        x_residual = x.clone()
        attention_out, attn_diag = self.attention(x, chaos_inject=True, z_value=gates[:, 3].item())
        attention_out = attention_out * attention_scale
        
        diagnostics_history.append(attn_diag)
        
        # Residual connection with temperature scaling
        x = x_residual + attention_out
        
        # Feed-forward with gating
        ffn_out = self.ffn1(x)
        ffn_out = torch.gelu(ffn_out) * ffn_scale
        ffn_out = self.ffn2(ffn_out)
        
        x = x + ffn_out
        
        return x, {'attn': attn_diag, 'gates': gates.mean(dim=0).tolist()}


class QwenThermodynamicModel(nn.Module):
    """
    Thermodynamically-enhanced Qwen model architecture.
    
    Integrates:
    - RoPE (Rotary Positional Embeddings) with temperature scaling
    - Attention mechanisms with entropy regularization
    - Feed-forward layers with chaos-based gating
    - Output head with thermodynamic constraints
    """
    
    def __init__(self, vocab_size: int = 151936, hidden_dim: int = 4096, 
                 num_heads: int = 32, num_layers: int = 16, max_seq_len: int = 8192):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # RoPE parameters
        self.rope_base = 10000.0
        self.rope_dims = num_heads * (hidden_dim // num_heads)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            ThermodynamicTransformerBlock(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Output head with thermodynamic constraints
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.entropy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, input_ids: torch.Tensor, 
                chaos_inject: bool = False, 
                z_values: Optional[torch.Tensor] = None,
                temperature_scale: float = 1.0):
        """
        Forward pass with thermodynamic learning.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            chaos_inject: Whether to inject chaos throughout the network
            z_values: Lorenz attractor Z values for each layer (optional)
            temperature_scale: Global temperature scaling factor
            
        Returns:
            Logits and thermodynamic diagnostics
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding(input_ids)  # (batch, seq_len, hidden_dim)
        
        # Add position embeddings with RoPE
        pos_emb = self.position_embedding(torch.arange(seq_len).unsqueeze(0))  # (1, seq_len, hidden_dim)
        x = x + pos_emb
        
        # Apply RoPE
        x = self._apply_rope(x)
        
        diagnostics_history = []
        
        for i, layer in enumerate(self.layers):
            if chaos_inject and z_values is not None:
                current_z = z_values[i].item() if i < len(z_values) else 0.0
            else:
                current_z = 0.0
            
            x, layer_diag = layer(x, diagnostics_history=diagnostics_history)
            
        # Output projection with temperature scaling
        logits = self.output_proj(x) / (temperature_scale + 1e-8)
        
        # Compute entropy for monitoring
        probs = torch.softmax(logits[:, -1], dim=-1)  # Last token only
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        
        diagnostics = {
            'entropy': entropy.item(),
            'temperature_scale': temperature_scale,
            'layer_diagnostics': diagnostics_history[-5:] if len(diagnostics_history) > 0 else [],
            'batch_size': batch_size,
            'seq_len': seq_len
        }
        
        return logits, diagnostics
    
    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Positional Embeddings."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute frequency embeddings
        inv_freq = 1.0 / (self.rope_base ** 
                          (torch.arange(0, self.rope_dims, 2).float() / self.rope_dims))
        
        # Position embeddings
        t = torch.arange(seq_len, device=x.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # Apply to hidden dimensions (rotate pairs)
        x_rope = x.view(batch_size, seq_len, self.num_heads, 
                       2, self.hidden_dim // (self.num_heads * 2))
        freqs = freqs.unsqueeze(0).unsqueeze(0)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        # Rotate pairs
        x_rope = torch.stack([
            x_rope[..., 0] * cos - x_rope[..., 1] * sin,
            x_rope[..., 0] * sin + x_rope[..., 1] * cos
        ], dim=-2)
        
        return x_rope.view(batch_size, seq_len, hidden_dim)


class QwenThermodynamicTrainer:
    """
    Trainer for thermodynamically-enhanced Qwen models.
    
    Implements:
    - Entropy regularization in loss function
    - Chaos-based curriculum learning
    - Temperature annealing schedule
    """
    
    def __init__(self, model: nn.Module, lr: float = 1e-4,
                 entropy_weight: float = 0.01, chaos_schedule: str = "linear",
                 device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Thermodynamic parameters
        self.entropy_weight = entropy_weight
        self.chaos_schedule = chaos_schedule
        
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    diagnostics: dict):
        """Compute loss with entropy regularization."""
        base_loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Entropy regularization term
        entropy = diagnostics['entropy']
        entropy_loss = self.entropy_weight * entropy
        
        return base_loss + entropy_loss
    
    def train_step(self, batch: dict, chaos_inject: bool = False):
        """Single training step with thermodynamic learning."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Get Z values if using chaos injection
        z_values = None
        if chaos_inject:
            from core.chaos import LorenzGenerator
            lorenz = LorenzGenerator()
            num_layers = len(self.model.layers) if hasattr(self.model, 'layers') else 16
            trajectory, _ = lorenz.step(num_layers)  # Generate one Z per layer
            z_values = torch.tensor(trajectory[:num_layers, 2])  # (num_layers,)
        
        with torch.no_grad():
            logits, diagnostics = self.model(input_ids, 
                                            chaos_inject=chaos_inject,
                                            z_values=z_values)
        
        labels_flat = labels.view(-1)
        loss = self.compute_loss(logits[:, -1], labels_flat, diagnostics)
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            **diagnostics
        }
