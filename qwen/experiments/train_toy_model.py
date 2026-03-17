#!/usr/bin/env python3
"""
Train a Toy Thermodynamic Model.

Goal: Train a small model on a simple corpus to enable qualitative comparison
      of sampling methods. We want the model to learn basic language structure
      so its output isn't just random noise.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the placeholder model (which is sufficient for this toy task)
from qwen.tests.test_qwen_thermodynamic import QwenThermodynamicModel

# A small corpus (Alice in Wonderland snippet)
CORPUS = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
There was nothing so VERY remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be too late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-POCKET, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.
"""

def train_toy_model():
    print("--- Training Toy Thermodynamic Model ---")
    
    # 1. Prepare Data
    # Simple character-level tokenizer
    chars = sorted(list(set(CORPUS)))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    
    print(f"Corpus length: {len(CORPUS)}")
    print(f"Vocab size: {vocab_size}")
    
    # Convert corpus to tensor
    data = torch.tensor([char_to_idx[c] for c in CORPUS], dtype=torch.long)
    
    # 2. Initialize Model
    device = torch.device('cpu')
    hidden_dim = 128
    model = QwenThermodynamicModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        max_seq_len=64
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    epochs = 200
    batch_size = 32
    seq_len = 32
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Random sampling of batches
        for _ in range(20): 
            start_idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            
            input_batch = torch.stack([data[i:i+seq_len] for i in start_idx]).to(device)
            target_batch = torch.stack([data[i+1:i+seq_len+1] for i in start_idx]).to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_batch)
            
            loss = criterion(logits.view(-1, vocab_size), target_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/num_batches:.4f}")
            
    # 4. Save Model
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/toy_model.pt'))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'hidden_dim': hidden_dim,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }, save_path)
    
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    train_toy_model()
