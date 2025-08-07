"""
Mixture of Experts (MoE) implementation for modern LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class Expert(nn.Module):
    """Single expert in MoE layer"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class Router(nn.Module):
    """Router to select experts"""
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts

    def forward(self, x):
        # Compute router logits
        router_logits = self.gate(x)
        return router_logits


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer"""
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        router_aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef

        # Create experts
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size) for _ in range(num_experts)
        ])
        
        # Router
        self.router = Router(hidden_size, num_experts)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Router forward pass
        router_logits = self.router(hidden_states_flat)
        
        # Apply softmax and get top-k experts
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize top-k weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            
            if expert_mask.sum() == 0:
                continue
                
            # Get tokens for this expert
            expert_tokens = hidden_states_flat[expert_mask]
            
            # Get weights for this expert
            expert_weights = topk_weights[expert_mask]
            expert_positions = (topk_indices[expert_mask] == expert_idx).float()
            expert_weights = (expert_weights * expert_positions).sum(dim=-1, keepdim=True)
            
            # Forward through expert
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Add weighted output
            final_hidden_states[expert_mask] += expert_weights * expert_output

        # Calculate auxiliary loss for load balancing
        aux_loss = self._calculate_aux_loss(routing_weights)

        # Reshape back to original shape
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)
        
        return final_hidden_states, aux_loss

    def _calculate_aux_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Calculate auxiliary loss for load balancing"""
        # Expert usage frequency
        expert_usage = routing_weights.sum(dim=0)
        expert_usage = expert_usage / expert_usage.sum()
        
        # Ideally, each expert should be used equally
        uniform_usage = torch.ones_like(expert_usage) / self.num_experts
        
        # Auxiliary loss to encourage balanced usage
        aux_loss = F.mse_loss(expert_usage, uniform_usage)
        
        return self.router_aux_loss_coef * aux_loss


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE layer"""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.moe = MixtureOfExperts(
            hidden_size, intermediate_size, num_experts, num_experts_per_tok
        )
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            attn_mask=attention_mask
        )
        hidden_states = residual + attn_output

        # MoE layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output, aux_loss = self.moe(hidden_states)
        hidden_states = residual + moe_output

        return hidden_states, aux_loss


if __name__ == "__main__":
    # Test MoE implementation
    print("Testing Mixture of Experts implementation...")
    
    # Parameters
    batch_size, seq_len, hidden_size = 2, 32, 512
    num_experts, num_experts_per_tok = 8, 2
    intermediate_size = hidden_size * 4
    
    # Create MoE layer
    moe = MixtureOfExperts(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    output, aux_loss = moe(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.6f}")
    
    # Test full transformer block
    moe_block = MoETransformerBlock(
        hidden_size=hidden_size,
        num_heads=8,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    
    block_output, block_aux_loss = moe_block(x)
    print(f"Transformer block output shape: {block_output.shape}")
    print(f"Transformer block aux loss: {block_aux_loss.item():.6f}")
    
    # Calculate parameter efficiency
    moe_params = sum(p.numel() for p in moe.parameters())
    dense_ffn = nn.Sequential(
        nn.Linear(hidden_size, intermediate_size),
        nn.ReLU(),
        nn.Linear(intermediate_size, hidden_size)
    )
    dense_params = sum(p.numel() for p in dense_ffn.parameters())
    
    print(f"\nParameter comparison:")
    print(f"MoE parameters: {moe_params:,}")
    print(f"Dense FFN parameters: {dense_params:,}")
    print(f"MoE uses {moe_params / dense_params:.1f}x more parameters")
    print(f"But only {num_experts_per_tok / num_experts:.1%} are active per token")
