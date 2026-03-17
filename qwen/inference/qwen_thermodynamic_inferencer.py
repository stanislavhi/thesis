#!/usr/bin/env python3
"""
Thermodynamically-Enhanced Qwen Inference.

Provides efficient inference with thermodynamic constraints, including:
- Temperature-based sampling with entropy awareness
- Chaos injection for exploration in uncertain regions
- Real-time monitoring of thermodynamic efficiency
"""

import torch
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for thermodynamically-enhanced inference."""
    max_length: int = 2048
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    use_chaos_injection: bool = False
    chaos_strength: float = 0.3
    entropy_threshold: float = 4.0


class QwenThermodynamicInferencer:
    """
    Inference engine for thermodynamically-enhanced Qwen models.
    
    Features:
    - Temperature-aware sampling with entropy constraints
    - Chaos injection for exploration in high-uncertainty regions
    - Beam search with thermodynamic penalties
    - Real-time monitoring of efficiency metrics
    """
    
    def __init__(self, model, config: Optional[InferenceConfig] = None):
        self.model = model.eval()
        
        if config is None:
            config = InferenceConfig()
            
        self.config = config
        
        # Setup device
        self.device = next(model.parameters()).device
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_length: Optional[int] = None,
                 temperature: Optional[float] = None,
                 chaos_inject: bool = False,
                 stream_output: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        Generate text with thermodynamic constraints.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len) or single sequence
            max_length: Maximum output length
            temperature: Sampling temperature
            chaos_inject: Whether to inject chaos during generation
            stream_output: Stream tokens as they're generated
            
        Returns:
            Generated tokens and diagnostics dict
        """
        if max_length is None:
            max_length = self.config.max_length
            
        if temperature is not None:
            self.config.temperature = temperature
        
        # Ensure input is on correct device
        input_ids = input_ids.to(self.device)
        
        batch_size, current_len = input_ids.shape
        
        diagnostics = {
            'total_entropy': 0.0,
            'temperature_history': [],
            'chaos_events': [],
            'avg_efficiency': 0.0
        }
        
        for step in range(current_len, max_length):
            # Forward pass with chaos injection if enabled
            with torch.no_grad():
                logits, layer_diag = self.model(input_ids, 
                                               chaos_inject=chaos_inject)
            
            # Extract last token logits
            last_logits = logits[:, -1]  # (batch_size, vocab_size)
            
            # Apply temperature scaling
            scaled_logits = last_logits / max(self.config.temperature, 1e-4)
            
            # Compute entropy for monitoring
            probs = torch.softmax(scaled_logits, dim=-1)
            local_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
            
            diagnostics['total_entropy'] += local_entropy
            diagnostics['temperature_history'].append(self.config.temperature)
            
            if chaos_inject:
                diagnostics['chaos_events'].append({
                    'step': step,
                    'entropy': local_entropy,
                    'temperature': self.config.temperature
                })
            
            # Thermodynamic-aware sampling
            next_tokens = self._sample_with_constraints(
                scaled_logits, 
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                entropy_threshold=self.config.entropy_threshold
            )
            
            # Append generated tokens
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        
        diagnostics['avg_efficiency'] = (
            max(0.1, 1 - diagnostics['total_entropy'] / (step + 1) * 2) 
            if 'step' in locals() else 1.0
        )
        
        return input_ids, diagnostics
    
    def _sample_with_constraints(self, logits: torch.Tensor, 
                                 top_k: int = 50, 
                                 top_p: float = 0.9,
                                 entropy_threshold: float = 4.0) -> torch.Tensor:
        """
        Sample with thermodynamic constraints using vectorized operations.
        """
        # 1. Thermodynamic Feedback (optional)
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        high_entropy_mask = entropy > entropy_threshold
        if high_entropy_mask.any():
            logits[high_entropy_mask] /= 1.5 

        # 2. Top-K Filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # 3. Top-P Filtering (Nucleus Sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # 4. Sample
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return next_tokens
    
    def beam_search(self, input_ids: torch.Tensor, beam_width: int = 5,
                    max_length: Optional[int] = None) -> Tuple[torch.Tensor, List[float]]:
        """
        Beam search with thermodynamic penalties.
        """
        if max_length is None:
            max_length = self.config.max_length
        
        device = self.device
        batch_size, current_len = input_ids.shape
        if batch_size != 1:
            raise NotImplementedError("Beam search currently supports batch_size=1 only")

        beams = [(0.0, input_ids, 0.0)] # (log_prob, sequence_tensor, total_entropy)
        
        for step in range(current_len, max_length):
            candidates = []
            
            # Stop if all beams have reached max length
            if all(b[1].shape[1] >= max_length for b in beams):
                break

            for score, seq, total_entropy in beams:
                with torch.no_grad():
                    logits, _ = self.model(seq)
                    last_logits = logits[:, -1, :]
                
                last_logits = last_logits / self.config.temperature
                probs = torch.softmax(last_logits, dim=-1)
                log_probs = torch.log(probs + 1e-8)
                
                step_entropy = -torch.sum(probs * log_probs).item()
                
                top_probs, top_indices = torch.topk(probs, beam_width)
                
                for i in range(beam_width):
                    token_idx = top_indices[0, i].item()
                    token_log_prob = log_probs[0, token_idx].item()
                    
                    new_score = score + token_log_prob
                    thermo_score = new_score + 0.1 * step_entropy
                    
                    new_seq = torch.cat([seq, torch.tensor([[token_idx]], device=device)], dim=-1)
                    candidates.append((thermo_score, new_seq, total_entropy + step_entropy))
            
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
        
        best_score, best_seq, _ = beams[0]
        return best_seq, [best_score]


class ThermodynamicSampler:
    """
    Specialized sampler for thermodynamically-enhanced generation.
    """
    
    def __init__(self, model, config: Optional[InferenceConfig] = None):
        self.model = model.eval()
        self.config = config or InferenceConfig()
        self.device = next(model.parameters()).device
    
    def sample(self, input_ids: torch.Tensor, 
               temperature_schedule: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        Sample with adaptive temperature and chaos injection.
        """
        batch_size, seq_len = input_ids.shape
        current_temp = self.config.temperature
        entropy_history = []
        
        for step in range(seq_len, self.config.max_length):
            with torch.no_grad():
                logits, layer_diag = self.model(input_ids, chaos_inject=self.config.use_chaos_injection)
            
            last_logits = logits[:, -1]
            
            probs_raw = torch.softmax(last_logits, dim=-1)
            local_entropy = -torch.sum(probs_raw * torch.log(probs_raw + 1e-8), dim=-1).mean().item()
            entropy_history.append(local_entropy)
            
            if temperature_schedule:
                target_temp = self.config.temperature
                if local_entropy > self.config.entropy_threshold:
                    target_temp *= 0.8
                elif local_entropy < 1.0:
                    target_temp *= 1.2
                current_temp = 0.9 * current_temp + 0.1 * target_temp
                current_temp = max(0.5, min(current_temp, 2.0))
            
            scaled_logits = last_logits / max(current_temp, 1e-4)
            
            # Use the same correct filtering logic
            if self.config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids, {
            'final_temperature': current_temp,
            'entropy_history': entropy_history,
            'avg_entropy': sum(entropy_history) / len(entropy_history) if entropy_history else 0.0
        }
