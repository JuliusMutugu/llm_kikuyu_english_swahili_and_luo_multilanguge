"""
Inference engine for modern LLMs with optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 100
    min_length: int = 1
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    pad_token_id: int = 0
    eos_token_id: int = 1
    use_cache: bool = True


class KVCache:
    """Key-Value cache for efficient generation"""
    def __init__(self, batch_size: int, num_heads: int, head_dim: int, max_length: int, device: torch.device):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.device = device
        
        # Initialize cache tensors
        self.key_cache = torch.zeros(
            batch_size, num_heads, max_length, head_dim, 
            device=device, dtype=torch.float16
        )
        self.value_cache = torch.zeros(
            batch_size, num_heads, max_length, head_dim, 
            device=device, dtype=torch.float16
        )
        self.cache_length = 0
    
    def update(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value pairs"""
        seq_len = key.size(2)
        
        # Store new key-value pairs
        self.key_cache[:, :, self.cache_length:self.cache_length + seq_len] = key
        self.value_cache[:, :, self.cache_length:self.cache_length + seq_len] = value
        
        # Update cache length
        self.cache_length += seq_len
        
        # Return full cached key-value pairs
        return (
            self.key_cache[:, :, :self.cache_length],
            self.value_cache[:, :, :self.cache_length]
        )
    
    def reset(self):
        """Reset cache for new generation"""
        self.cache_length = 0


class OptimizedInferenceEngine:
    """Optimized inference engine for LLMs"""
    
    def __init__(self, model: nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Enable optimizations if available
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
    
    def prepare_inputs(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Prepare inputs for inference"""
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def generate_greedy(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 100,
        use_cache: bool = True
    ) -> torch.Tensor:
        """Greedy decoding generation"""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        # Initialize KV cache if using
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length - seq_len):
                # Forward pass
                if use_cache and past_key_values is not None:
                    # Only use last token for generation
                    outputs = self.model(
                        generated[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                else:
                    outputs = self.model(generated, use_cache=use_cache)
                
                # Extract logits and cache
                if isinstance(outputs, tuple):
                    logits, past_key_values = outputs
                else:
                    logits = outputs
                    past_key_values = None
                
                # Get next token
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return generated
    
    def generate_nucleus(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """Nucleus (top-p) sampling generation"""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        past_key_values = None
        
        with torch.no_grad():
            for step in range(config.max_length - seq_len):
                # Forward pass
                if config.use_cache and past_key_values is not None:
                    outputs = self.model(
                        generated[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                else:
                    outputs = self.model(generated, use_cache=config.use_cache)
                
                if isinstance(outputs, tuple):
                    logits, past_key_values = outputs
                else:
                    logits = outputs
                    past_key_values = None
                
                # Apply temperature
                next_token_logits = logits[:, -1, :] / config.temperature
                
                # Apply repetition penalty
                if config.repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, generated, config.repetition_penalty
                    )
                
                # Apply top-k filtering
                if config.top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                
                # Apply nucleus (top-p) filtering
                if config.top_p < 1.0:
                    next_token_logits = self._nucleus_filtering(next_token_logits, config.top_p)
                
                # Sample next token
                if config.do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == config.eos_token_id:
                    break
        
        return generated
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated: torch.Tensor, 
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        batch_size, vocab_size = logits.shape
        
        for batch_idx in range(batch_size):
            for token_id in set(generated[batch_idx].tolist()):
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits to only keep top k tokens"""
        if top_k <= 0:
            return logits
        
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _nucleus_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus filtering (top-p)"""
        if top_p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted indices to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def generate_text(
        self, 
        prompt: str, 
        config: GenerationConfig = None
    ) -> str:
        """Generate text from prompt"""
        if config is None:
            config = GenerationConfig()
        
        # Prepare inputs
        inputs = self.prepare_inputs(prompt, max_length=512)
        input_ids = inputs['input_ids']
        
        # Generate
        start_time = time.time()
        
        if config.do_sample:
            generated_ids = self.generate_nucleus(input_ids, config)
        else:
            generated_ids = self.generate_greedy(
                input_ids, 
                max_length=config.max_length,
                use_cache=config.use_cache
            )
        
        generation_time = time.time() - start_time
        
        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )
        
        # Calculate metrics
        num_tokens = generated_ids.shape[1] - input_ids.shape[1]
        tokens_per_second = num_tokens / generation_time
        
        print(f"Generated {num_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
        
        return generated_text
    
    def benchmark_inference(self, prompt: str, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark inference performance"""
        print(f"Benchmarking inference with {num_runs} runs...")
        
        inputs = self.prepare_inputs(prompt)
        input_ids = inputs['input_ids']
        
        times = []
        token_counts = []
        
        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.time()
                
                # Generate with greedy decoding
                generated = self.generate_greedy(input_ids, max_length=50, use_cache=True)
                
                end_time = time.time()
                
                times.append(end_time - start_time)
                token_counts.append(generated.shape[1] - input_ids.shape[1])
        
        # Calculate statistics
        avg_time = np.mean(times)
        avg_tokens = np.mean(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        results = {
            'avg_time_s': avg_time,
            'avg_tokens': avg_tokens,
            'tokens_per_second': tokens_per_second,
            'min_time_s': np.min(times),
            'max_time_s': np.max(times),
            'std_time_s': np.std(times)
        }
        
        return results


if __name__ == "__main__":
    # Example usage (requires a model and tokenizer)
    print("Inference engine implementation completed!")
    print("To use:")
    print("1. Load your model and tokenizer")
    print("2. Create OptimizedInferenceEngine(model, tokenizer, device)")
    print("3. Use generate_text() for text generation")
    print("4. Use benchmark_inference() for performance testing")
