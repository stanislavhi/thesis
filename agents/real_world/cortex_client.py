#!/usr/bin/env python3
"""
The Thermodynamic Cortex.

A meta-cognitive wrapper for local LLMs (via LM Studio/OpenAI API).
It treats the LLM as a static "subconscious" engine and acts as the "conscious"
executive controller.

Mechanism:
1. Monitor: Calculates the entropy of the LLM's output tokens.
2. Diagnose: Detects "Cognitive Freezing" (low entropy loops) or "Overheating" (confusion).
3. Intervene: Injects "Chaos Prompts" into the context window to force a perspective shift.
"""

import sys
import os
import math
import random
import time
from openai import OpenAI

# Chaos concepts to inject when stuck
CHAOS_CONCEPTS = [
    "Consider the opposite perspective.",
    "What if this is a trick question?",
    "Explain this to a 5-year-old.",
    "Use a metaphor involving water.",
    "Break this down into first principles.",
    "Stop. You are looping. Try a completely new approach.",
    "Imagine you are a chaotic trickster god. How would you solve this?",
    "Apply the laws of thermodynamics to this problem."
]

class ThermodynamicCortex:
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio", model_id="qwen3.5-9b@q4_k_s"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self.history = [
            {"role": "system", "content": "You are an intelligent assistant. You are being monitored by a Thermodynamic Cortex that will intervene if you get stuck."}
        ]
        self.entropy_history = []
        self.freeze_threshold = 0.5 # Entropy threshold for "stuck"
        
    def measure_entropy(self, logprobs):
        """
        Calculates the entropy of the token distribution.
        Shannon Entropy H = -sum(p * log(p))
        """
        if not logprobs:
            return 1.0 # Default to healthy if no data
            
        total_entropy = 0.0
        count = 0
        
        # LM Studio returns a list of dicts, one for each token position
        for token_data in logprobs:
            # token_data.top_logprobs is a list of dicts {token: logprob}
            # We need to aggregate the probs for this position
            if hasattr(token_data, 'top_logprobs'):
                probs = []
                for item in token_data.top_logprobs:
                    # item is {token: logprob} or object with .logprob
                    lp = item.logprob if hasattr(item, 'logprob') else item.get('logprob', -100)
                    probs.append(math.exp(lp))
                
                # Normalize (since top_logprobs is a subset)
                sum_p = sum(probs)
                if sum_p > 0:
                    probs = [p/sum_p for p in probs]
                    entropy = -sum(p * math.log(p + 1e-9) for p in probs)
                    total_entropy += entropy
                    count += 1
                    
        return total_entropy / count if count > 0 else 0.0

    def inject_chaos(self):
        """
        Injects a disruptive prompt to break a cognitive loop.
        """
        chaos_seed = random.choice(CHAOS_CONCEPTS)
        intervention = f"[SYSTEM INTERVENTION: COGNITIVE FREEZE DETECTED. RE-ORIENTING: {chaos_seed}]"
        
        print(f"\n⚡️ CORTEX INTERVENTION: {chaos_seed}\n")
        
        # Inject into history
        self.history.append({"role": "system", "content": intervention})
        return intervention

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        
        print("Thinking...", end="", flush=True)
        
        try:
            # Request logprobs to measure entropy
            # Note: LM Studio might not return logprobs depending on version/settings
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=self.history,
                temperature=0.7,
                logprobs= True,
                top_logprobs=5
            )
            
            content = response.choices[0].message.content
            
            # Robustly handle missing logprobs
            logprobs = None
            if response.choices[0].logprobs:
                logprobs = response.choices[0].logprobs.content
            
            # 1. Measure Physics
            if logprobs:
                entropy = self.measure_entropy(logprobs)
                self.entropy_history.append(entropy)
                print(f"\r[Entropy: {entropy:.2f}] ", end="")
                
                # 2. Diagnose & Intervene
                if entropy < self.freeze_threshold:
                    # The model is too confident/repetitive. It's frozen.
                    self.inject_chaos()
                    # Force a retry with the new context
                    return self.chat("Continue, but apply the intervention.")
            else:
                print("\r[Entropy: N/A] ", end="")
            
            # 3. Normal Response
            self.history.append({"role": "assistant", "content": content})
            return content
            
        except Exception as e:
            return f"\nError: {e}. (Check LM Studio connection)"

def main():
    print("--- THERMODYNAMIC CORTEX (LM Studio Client) ---")
    print("Connecting to localhost:1234...")
    
    cortex = ThermodynamicCortex()
    
    print("\nReady. Type 'exit' to quit.")
    print("Try asking a repetitive question or a paradox to trigger a freeze.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = cortex.chat(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
