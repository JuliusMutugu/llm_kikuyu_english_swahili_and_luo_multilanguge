"""
Conversational Mixture of Experts Model
Simple, laptop-friendly implementation for multilingual conversations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

class ConversationExpert(nn.Module):
    """Specialized expert for different conversation aspects"""
    def __init__(self, hidden_size: int, expert_type: str = "general"):
        super().__init__()
        self.expert_type = expert_type
        self.hidden_size = hidden_size
        
        # Smaller intermediate size for laptop efficiency
        intermediate_size = hidden_size * 2
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.dropout(self.down_proj(F.gelu(gate) * up))

class ConversationRouter(nn.Module):
    """Smart router for conversation contexts"""
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Context-aware routing
        self.context_embedding = nn.Linear(hidden_size, hidden_size // 4)
        self.router_gate = nn.Linear(hidden_size // 4, num_experts, bias=False)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x, conversation_context=None):
        # Simple routing based on input embeddings
        context_emb = F.gelu(self.context_embedding(x))
        router_logits = self.router_gate(context_emb) / self.temperature
        return router_logits

class ConversationalMoE(nn.Module):
    """Laptop-friendly Conversational Mixture of Experts"""
    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_size: int = 256,  # Smaller for laptop efficiency
        num_experts: int = 4,    # Fewer experts for simplicity
        num_experts_per_token: int = 2,
        max_seq_length: int = 128,
        num_layers: int = 4,     # Fewer layers
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        # Expert specializations for conversation
        expert_types = ["greeting", "question", "response", "general"]
        self.experts = nn.ModuleList([
            ConversationExpert(hidden_size, expert_types[i % len(expert_types)]) 
            for i in range(num_experts)
        ])
        
        # Router
        self.router = ConversationRouter(hidden_size, num_experts)
        
        # Simple attention for conversation context
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True, dropout=0.1
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # Conversation state tracking
        self.conversation_memory = None

    def forward(self, input_ids, attention_mask=None, conversation_context=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        hidden_states = token_emb + pos_emb
        
        # Self-attention for conversation understanding
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # MoE processing
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        moe_output, routing_info = self._moe_forward(hidden_states, conversation_context)
        hidden_states = residual + moe_output
        
        # Output projection
        logits = self.output_proj(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'routing_info': routing_info
        }
    
    def _moe_forward(self, hidden_states, conversation_context=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        
        # Get routing decisions
        router_logits = self.router(hidden_flat, conversation_context)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.num_experts_per_token, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output
        final_output = torch.zeros_like(hidden_flat)
        
        # Process with selected experts
        for expert_idx in range(self.num_experts):
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            
            if expert_mask.sum() == 0:
                continue
            
            # Get tokens for this expert
            expert_tokens = hidden_flat[expert_mask]
            expert_weights = topk_weights[expert_mask]
            expert_positions = (topk_indices[expert_mask] == expert_idx).float()
            weights = (expert_weights * expert_positions).sum(dim=-1, keepdim=True)
            
            # Forward through expert
            expert_output = self.experts[expert_idx](expert_tokens)
            final_output[expert_mask] += weights * expert_output
        
        # Reshape back
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        routing_info = {
            'routing_weights': routing_weights,
            'expert_usage': routing_weights.sum(dim=0) / routing_weights.sum()
        }
        
        return final_output, routing_info

class ConversationalMoESystem:
    """Complete conversational system with MoE integration"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Simple tokenizer (character-level for multilingual support)
        self.vocab = self._build_multilingual_vocab()
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # Initialize model
        self.model = ConversationalMoE(
            vocab_size=self.vocab_size,
            hidden_size=256,
            num_experts=4,
            num_experts_per_token=2,
            max_seq_length=128,
            num_layers=4
        ).to(self.device)
        
        # Conversation state
        self.conversation_history = []
        self.conversation_context = None
        
        if model_path:
            self.load_model(model_path)
    
    def _build_multilingual_vocab(self):
        """Build vocabulary supporting multiple languages"""
        # Basic characters
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        chars.extend([" ", ".", ",", "?", "!", "'", '"', "-", "(", ")", "\n"])
        
        # Kiswahili specific
        chars.extend(["ã", "ĩ", "ũ", "ẽ", "õ"])
        
        # Kikuyu specific  
        chars.extend(["ĩ", "ũ", "ẽ", "ĩ"])
        
        # Common punctuation and symbols
        chars.extend([":", ";", "@", "#", "$", "%", "&", "*", "+", "=", "[", "]", "{", "}", "|", "\\", "/"])
        
        # Add special tokens
        special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]
        
        return special_tokens + list(set(chars))
    
    def tokenize(self, text: str, max_length: int = 128) -> List[int]:
        """Simple character-level tokenization"""
        tokens = [self.char_to_idx.get("<START>", 0)]
        
        for char in text[:max_length-2]:  # Leave space for START/END
            tokens.append(self.char_to_idx.get(char, self.char_to_idx.get("<UNK>", 1)))
        
        tokens.append(self.char_to_idx.get("<END>", 3))
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.char_to_idx.get("<PAD>", 0))
            
        return tokens[:max_length]
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text"""
        chars = []
        for token in tokens:
            char = self.idx_to_char.get(token, "")
            if char in ["<PAD>", "<START>", "<END>"]:
                continue
            if char == "<UNK>":
                char = "?"
            chars.append(char)
        return "".join(chars).strip()
    
    def generate_response(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 0.8,
        conversation_id: str = None
    ) -> Dict:
        """Generate conversational response using MoE"""
        self.model.eval()
        
        # Update conversation context
        self.conversation_history.append({"role": "user", "content": prompt})
        
        # Tokenize input
        input_tokens = self.tokenize(prompt)
        input_ids = torch.tensor([input_tokens], device=self.device)
        
        generated_tokens = []
        current_input = input_ids
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.model(current_input, conversation_context=self.conversation_context)
                logits = outputs['logits']
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token.item())
                
                # Check for end token
                if next_token.item() == self.char_to_idx.get("<END>", 3):
                    break
                
                # Update input for next iteration
                current_input = torch.cat([
                    current_input, 
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                
                # Keep only recent context to manage memory
                if current_input.size(1) > 128:
                    current_input = current_input[:, -64:]
        
        # Decode response
        response_text = self.detokenize(generated_tokens)
        
        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Extract routing information for insights
        routing_info = outputs.get('routing_info', {})
        expert_usage = routing_info.get('expert_usage', torch.zeros(4))
        
        return {
            "response": response_text,
            "conversation_id": conversation_id or f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "tokens_generated": len(generated_tokens),
            "expert_usage": expert_usage.tolist() if torch.is_tensor(expert_usage) else expert_usage,
            "conversation_turn": len(self.conversation_history)
        }
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Could not load model: {e}")
    
    def get_conversation_stats(self):
        """Get conversation statistics"""
        total_turns = len(self.conversation_history)
        user_turns = sum(1 for msg in self.conversation_history if msg['role'] == 'user')
        assistant_turns = sum(1 for msg in self.conversation_history if msg['role'] == 'assistant')
        
        return {
            "total_turns": total_turns,
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "conversation_length": len(str(self.conversation_history))
        }

def test_conversational_moe():
    """Test the conversational MoE system"""
    print("🤖 Testing Conversational MoE System...")
    
    # Initialize system
    moe_system = ConversationalMoESystem()
    
    # Test conversations in different languages
    test_prompts = [
        "Hello, how are you today?",
        "Habari yako? Unahali aje?",
        "Wĩ atĩa? Ũrĩ na thayũ?",
        "What can you help me with?",
        "Unaweza kunisaidia nini?"
    ]
    
    print("\n📝 Testing multilingual conversations:")
    for i, prompt in enumerate(test_prompts):
        print(f"\n🔹 Test {i+1}: {prompt}")
        
        result = moe_system.generate_response(
            prompt, 
            max_length=50,
            temperature=0.8
        )
        
        print(f"🤖 Response: {result['response']}")
        print(f"📊 Tokens: {result['tokens_generated']}")
        print(f"🎯 Expert usage: {[f'{x:.2f}' for x in result['expert_usage']]}")
    
    # Show conversation stats
    stats = moe_system.get_conversation_stats()
    print(f"\n📈 Conversation Statistics:")
    print(f"   Total turns: {stats['total_turns']}")
    print(f"   User turns: {stats['user_turns']}")
    print(f"   Assistant turns: {stats['assistant_turns']}")
    
    # Show model size
    total_params = sum(p.numel() for p in moe_system.model.parameters())
    trainable_params = sum(p.numel() for p in moe_system.model.parameters() if p.requires_grad)
    
    print(f"\n🔧 Model Information:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return moe_system

if __name__ == "__main__":
    # Test the system
    moe_system = test_conversational_moe()
    
    # Save the initialized model
    moe_system.save_model("conversational_moe_model.pt")
    print("\n✅ Conversational MoE system ready!")
