"""
Simplified but Effective LLM Model for Better Training Performance
Optimized for trilingual East African languages with improved architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SimpleConfig:
    """Configuration for simplified but effective LLM"""
    vocab_size: int = 8000
    hidden_size: int = 512        # Balanced size for good performance
    num_layers: int = 8           # More layers for better understanding
    num_heads: int = 8            # Multi-head attention
    num_kv_heads: int = 4         # Multi-query attention for efficiency
    intermediate_size: int = 2048  # FFN size
    max_length: int = 512         # Maximum sequence length
    dropout: float = 0.1          # Dropout for regularization
    layer_norm_epsilon: float = 1e-5
    use_rotary_emb: bool = True   # Rotary position embeddings
    use_flash_attention: bool = False  # Disable for compatibility
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True  # Share input/output embeddings


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - more stable than LayerNorm"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding for better positional understanding"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute rotary embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int):
        # x shape: [batch_size, seq_len, num_heads, head_dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Create rotation matrix
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        
        return cos, sin
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # Rotate half of the features
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention for efficiency - shared key/value heads"""
    
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Rotary embeddings
        if config.use_rotary_emb:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim, 
                config.max_length, 
                config.rope_theta
            )
        else:
            self.rotary_emb = None
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to queries, keys, values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(query_states, seq_len)
            query_states = self.rotary_emb.apply_rotary_pos_emb(query_states, cos, sin)
            key_states = self.rotary_emb.apply_rotary_pos_emb(key_states, cos, sin)
        
        # Expand key/value states for multi-query attention
        if self.num_kv_heads != self.num_heads:
            key_states = key_states.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            value_states = value_states.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        # Apply causal mask
        if attention_mask is not None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class SwiGLU(nn.Module):
    """SwiGLU activation function - better than ReLU for language models"""
    
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Enhanced Transformer Block with modern improvements"""
    
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.attention = MultiQueryAttention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.ffn_norm = RMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Pre-norm attention
        normed_hidden_states = self.attention_norm(hidden_states)
        attn_output = self.attention(normed_hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Pre-norm feed-forward
        normed_hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.feed_forward(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(ffn_output)
        
        return hidden_states


class SimpleLLM(nn.Module):
    """Simplified but effective LLM for trilingual conversation"""
    
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final norm and output
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_epsilon)
        
        if config.tie_word_embeddings:
            # Share input and output embeddings
            self.lm_head = lambda x: F.linear(x, self.embed_tokens.weight)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Output projection
        logits = self.lm_head(hidden_states)
        
        return ModelOutput(last_hidden_state=logits, hidden_states=hidden_states)
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 100, 
        temperature: float = 0.8,
        top_k: int = 50,
        do_sample: bool = True
    ):
        """Generate text with improved sampling"""
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                if generated.size(1) >= self.config.max_length:
                    break
                
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs.last_hidden_state[:, -1, :]
                
                if do_sample:
                    # Temperature scaling
                    logits = logits / temperature
                    
                    # Top-k sampling
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        probs = F.softmax(top_k_logits, dim=-1)
                        next_token_idx = torch.multinomial(probs, 1)
                        next_token = top_k_indices.gather(-1, next_token_idx)
                    else:
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                else:
                    # Greedy sampling
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class ModelOutput:
    """Simple model output class"""
    def __init__(self, last_hidden_state, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


def create_model_and_tokenizer(vocab_size: int = 8000):
    """Create model and tokenizer for training"""
    config = SimpleConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=4,
        intermediate_size=2048,
        max_length=512,
        dropout=0.1,
        use_rotary_emb=True,
        tie_word_embeddings=True
    )
    
    model = SimpleLLM(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Memory usage: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.1f} MB")
    
    return model, config


if __name__ == "__main__":
    # Test model creation
    model, config = create_model_and_tokenizer()
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"Output shape: {outputs.last_hidden_state.shape}")
        
        # Test generation
        generated = model.generate(input_ids[:1, :10], max_length=50)
        print(f"Generated shape: {generated.shape}")
    
    print("✅ Model test passed!")
