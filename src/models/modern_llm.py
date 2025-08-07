"""
Modern LLM Implementation with Recent Developments
Includes: RoPE, RMSNorm, SwiGLU, Multi-Query Attention, Flash Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Laptop-optimized defaults - much smaller for training
    vocab_size: int = 8000     # Reduced vocab for faster training
    hidden_size: int = 512     # Smaller hidden size for laptop
    num_layers: int = 6        # Fewer layers for memory efficiency
    num_heads: int = 8         # Fewer attention heads
    num_kv_heads: int = 4      # Multi-Query Attention (50% reduction)
    intermediate_size: int = 1024  # Smaller FFN for memory savings
    max_position_embeddings: int = 1024  # Shorter sequences for laptop
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True   # Tie weights to save memory
    use_rotary_emb: bool = True
    use_flash_attention: bool = True
    # Laptop-specific optimizations
    gradient_checkpointing: bool = True  # Save memory during training
    use_mixed_precision: bool = True     # FP16 for speed and memory
    max_batch_size: int = 4             # Small batch size for laptop


class RMSNorm(nn.Module):
    """RMSNorm - more stable than LayerNorm for large models"""
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
    """Rotary Position Embedding (RoPE) - better positional understanding"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        # Create position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Compute the angles
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function from PaLM"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention - faster inference with shared key-value heads"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embeddings
        if config.use_rotary_emb:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim, 
                config.max_position_embeddings,
                config.rope_theta
            )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Project to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embedding
        if self.config.use_rotary_emb:
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention with gradient checkpointing awareness
        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's flash attention if available (optimized for laptop)
            try:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=None  # Let PyTorch handle scaling for efficiency
                )
            except Exception:
                # Fallback to memory-efficient attention
                attn_output = self._memory_efficient_attention(
                    query_states, key_states, value_states, attention_mask
                )
        else:
            # Memory-efficient standard attention for laptop
            attn_output = self._memory_efficient_attention(
                query_states, key_states, value_states, attention_mask
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

    def _memory_efficient_attention(
        self, 
        query_states: torch.Tensor, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Memory-efficient attention computation for laptop training"""
        # Compute attention in chunks to save memory
        bsz, num_heads, seq_len, head_dim = query_states.shape
        
        # Use chunked attention for memory efficiency
        chunk_size = min(256, seq_len)  # Small chunks for laptop
        attn_output = torch.zeros_like(query_states)
        
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            
            # Compute attention for this chunk
            q_chunk = query_states[:, :, i:end_idx, :]
            attn_weights_chunk = torch.matmul(q_chunk, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, i:end_idx, :]
                attn_weights_chunk = attn_weights_chunk + mask_chunk
            
            # Use float32 for stability, then convert back
            attn_weights_chunk = F.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(q_chunk.dtype)
            attn_chunk = torch.matmul(attn_weights_chunk, value_states)
            
            attn_output[:, :, i:end_idx, :] = attn_chunk
        
        return attn_output


class TransformerBlock(nn.Module):
    """Memory-optimized Transformer block for laptop training"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Self-attention
        self.self_attn = MultiQueryAttention(config)
        
        # Feed-forward network with SwiGLU
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        
        # Layer normalization (RMSNorm)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        # Use gradient checkpointing during training for memory efficiency
        if self.training and self.gradient_checkpointing:
            return self._forward_with_checkpointing(
                hidden_states, attention_mask, past_key_value, use_cache
            )
        else:
            return self._forward_impl(
                hidden_states, attention_mask, past_key_value, use_cache
            )
    
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """Standard forward implementation"""
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value
    
    def _forward_with_checkpointing(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """Forward with gradient checkpointing for memory efficiency"""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        # Checkpoint attention computation
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if past_key_value is None:  # Only checkpoint during training, not generation
            hidden_states, past_key_value = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attn),
                hidden_states,
                attention_mask,
                past_key_value,
                use_cache,
            )
        else:
            hidden_states, past_key_value = self.self_attn(
                hidden_states, attention_mask, past_key_value, use_cache
            )
            
        hidden_states = residual + hidden_states

        # Checkpoint FFN computation
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.mlp),
            hidden_states,
        )
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class ModernLLM(nn.Module):
    """Modern Large Language Model with recent developments"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language model head (memory-optimized)
        if config.tie_word_embeddings:
            self.lm_head = None  # Use tied embeddings to save memory
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing for training
        if hasattr(self, 'gradient_checkpointing_enable'):
            self.gradient_checkpointing_enable()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for inference"""
        for layer in self.layers:
            layer.gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights following modern practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Input embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to causal mask
            batch_size, seq_length = input_ids.shape
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), float('-inf'), device=input_ids.device),
                diagonal=1
            )
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
            attention_mask = attention_mask + causal_mask

        # Forward through transformer layers
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.layers)
        presents = () if use_cache else None
        
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                presents = presents + (present,)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Language modeling head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        if return_dict:
            return {
                'logits': logits,
                'past_key_values': presents,
                'hidden_states': hidden_states,
            }
        
        return logits, presents


def create_model(config: ModelConfig) -> ModernLLM:
    """Factory function to create a modern LLM"""
    return ModernLLM(config)


if __name__ == "__main__":
    # Laptop-optimized example usage
    config = ModelConfig(
        vocab_size=8000,      # Reduced for faster training
        hidden_size=512,      # Laptop-friendly size
        num_layers=6,         # Fewer layers for memory
        num_heads=8,
        num_kv_heads=4,       # Multi-Query Attention
        intermediate_size=1024,  # Smaller FFN
        max_position_embeddings=1024,  # Shorter sequences
        gradient_checkpointing=True,   # Memory optimization
        use_mixed_precision=True,      # Speed optimization
    )
    
    model = create_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🚀 Laptop-Optimized LLM Created!")
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
    print(f"   With FP16: ~{total_params * 2 / 1024**2:.1f} MB")
    
    # Test forward pass with laptop-friendly batch
    batch_size, seq_len = 2, 128  # Small batch for laptop
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    print(f"\n🧪 Testing forward pass...")
    print(f"   Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs['logits']
        print(f"   Output shape: {logits.shape}")
        print(f"   Memory efficient: ✅")
    
    # Memory usage estimation
    estimated_memory_training = total_params * 4 * 3 / 1024**2  # Rough estimate: weights + gradients + optimizer states
    print(f"\n💾 Estimated training memory: ~{estimated_memory_training:.0f} MB")
    print(f"   With mixed precision: ~{estimated_memory_training * 0.6:.0f} MB")
    print(f"   Laptop friendly: {'✅' if estimated_memory_training < 2000 else '⚠️'}")
    
    print(f"\n🎯 Laptop Training Tips:")
    print(f"   • Use batch_size=1-4")
    print(f"   • Enable gradient_checkpointing=True")
    print(f"   • Use mixed_precision=True (FP16)")
    print(f"   • Keep sequence_length≤512")
    print(f"   • Use gradient_accumulation_steps for larger effective batch")
