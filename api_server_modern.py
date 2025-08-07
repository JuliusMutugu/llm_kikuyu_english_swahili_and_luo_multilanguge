#!/usr/bin/env python3
"""
Modern API Server for Trilingual LLM
Optimized for Streamlit interface with enhanced features
"""

import os
import sys
import json
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(default="auto", pattern="^(auto|en|sw|ki|luo)$")
    conversation_id: Optional[str] = None
    max_length: int = Field(default=100, ge=10, le=500)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    top_k: int = Field(default=30, ge=1, le=100)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    language_detected: str
    confidence: float
    tokens_generated: int
    processing_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    memory_usage: Dict[str, float]
    api_version: str

# Conversation storage
conversations: Dict[str, List[Dict]] = {}

# Simple tokenizer for character-level processing
class SimpleTokenizer:
    def __init__(self):
        # Common characters in all supported languages
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        # Basic character set (will be expanded based on training data)
        basic_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\"")
        
        # Add common characters from supported languages
        swahili_chars = list("âêîôûñç")
        kikuyu_chars = list("ĩũĩĩũĩũĩũ")  # Common Kikuyu diacritics
        
        all_chars = basic_chars + swahili_chars + kikuyu_chars
        
        # Remove duplicates and sort
        unique_chars = sorted(list(set(all_chars)))
        
        self.vocab = self.special_tokens + unique_chars
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.vocab_size = len(self.vocab)
        
        logger.info(f"Tokenizer initialized with {self.vocab_size} tokens")
    
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        tokens = [self.bos_token_id]
        
        for char in text:
            token_id = self.char_to_idx.get(char, self.unk_token_id)
            tokens.append(token_id)
        
        tokens.append(self.eos_token_id)
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        
        # Pad if needed
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        chars = []
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            char = self.idx_to_char.get(token_id, '<UNK>')
            if char != '<UNK>':
                chars.append(char)
        return ''.join(chars)

# Simple model for demonstration
class SimpleLLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, num_layers: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        
        # Simple transformer-like architecture
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        # Create attention mask
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Convert to boolean and invert for transformer
        src_key_padding_mask = ~attention_mask.bool()
        
        # Embeddings
        x = self.embedding(input_ids)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output projection
        logits = self.lm_head(x)
        
        return type('ModelOutput', (), {'logits': logits})()

# Global model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load the trained model"""
    global model, tokenizer, device
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Try to load trained model
    checkpoint_path = Path("checkpoints/final_model.pt")
    
    if checkpoint_path.exists():
        logger.info("Loading trained model...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Create tokenizer from saved vocab
            tokenizer = SimpleTokenizer()
            if 'tokenizer' in checkpoint:
                tokenizer.char_to_idx = checkpoint['tokenizer'].get('char_to_idx', tokenizer.char_to_idx)
                tokenizer.idx_to_char = checkpoint['tokenizer'].get('idx_to_char', tokenizer.idx_to_char)
                tokenizer.vocab_size = checkpoint['tokenizer'].get('vocab_size', tokenizer.vocab_size)
            
            # Create model
            model = SimpleLLM(vocab_size=tokenizer.vocab_size, hidden_size=128, num_layers=3)
            
            # Load state dict if available
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Trained model loaded successfully")
            else:
                logger.warning("⚠️ No model state dict found, using untrained model")
            
        except Exception as e:
            logger.error(f"❌ Error loading trained model: {e}")
            # Fall back to fresh model
            tokenizer = SimpleTokenizer()
            model = SimpleLLM(vocab_size=tokenizer.vocab_size, hidden_size=128, num_layers=3)
    else:
        logger.info("No trained model found, creating fresh model...")
        tokenizer = SimpleTokenizer()
        model = SimpleLLM(vocab_size=tokenizer.vocab_size, hidden_size=128, num_layers=3)
    
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded with {total_params:,} parameters")

def detect_language(text: str) -> tuple[str, float]:
    """Simple language detection"""
    text_lower = text.lower()
    
    # English indicators
    english_words = ['the', 'and', 'is', 'are', 'you', 'me', 'my', 'we', 'he', 'she', 'it', 'can', 'will', 'what', 'how', 'when', 'where']
    english_score = sum(1 for word in english_words if word in text_lower)
    
    # Swahili indicators
    swahili_words = ['na', 'ni', 'ya', 'wa', 'za', 'la', 'ku', 'mimi', 'wewe', 'yeye', 'sisi', 'nyinyi', 'wao', 'habari', 'asante', 'karibu']
    swahili_score = sum(1 for word in swahili_words if word in text_lower)
    
    # Kikuyu indicators (basic)
    kikuyu_words = ['nĩ', 'wa', 'na', 'ũ', 'mũ', 'kĩ', 'rĩ', 'wĩ', 'atĩa', 'mwega', 'ngai']
    kikuyu_score = sum(1 for word in kikuyu_words if word in text_lower)
    
    # Luo indicators (basic)
    luo_words = ['to', 'gi', 'ne', 'ka', 'ma', 'jo', 'nyar', 'wuon', 'inadi', 'nade', 'maber']
    luo_score = sum(1 for word in luo_words if word in text_lower)
    
    # Determine language
    scores = {
        'English': english_score,
        'Kiswahili': swahili_score,
        'Kikuyu': kikuyu_score,
        'Luo': luo_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'English', 0.5  # Default to English with low confidence
    
    detected_lang = max(scores, key=scores.get)
    confidence = min(max_score / (len(text.split()) + 1), 1.0)
    
    return detected_lang, confidence

def generate_response(prompt: str, max_length: int = 100, temperature: float = 0.7, top_k: int = 30) -> tuple[str, int]:
    """Generate response using the model"""
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        return "Model not loaded. Please restart the server.", 0
    
    try:
        # Encode prompt
        input_ids = torch.tensor(
            tokenizer.encode(prompt, max_length=50),
            dtype=torch.long
        ).unsqueeze(0).to(device)
        
        generated = input_ids.clone()
        tokens_generated = 0
        
        model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                if generated.size(1) >= 200:  # Prevent too long sequences
                    break
                
                # Forward pass
                outputs = model(generated)
                logits = outputs.logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1)
                    next_token = top_k_indices.gather(-1, next_token_idx)
                else:
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                
                # Add to sequence
                generated = torch.cat([generated, next_token], dim=1)
                tokens_generated += 1
                
                # Stop at EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        
        # Clean up the response
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Ensure we have some response
        if not generated_text or len(generated_text.strip()) < 3:
            # Fallback responses based on language detection
            lang, _ = detect_language(prompt)
            fallback_responses = {
                'English': "Thank you for your message! I'm learning to respond better in multiple languages.",
                'Kiswahili': "Asante kwa ujumbe wako! Ninajifunza kujibu vizuri kwa lugha nyingi.",
                'Kikuyu': "Nĩngũgũcookeria ngaatho! Ndĩ gũthoma kũnjookia wega na ciũgano nyingĩ.",
                'Luo': "Erokamano kuom otena! Apuonjora dwaro maber e dhok mopogore opogore."
            }
            generated_text = fallback_responses.get(lang, fallback_responses['English'])
            tokens_generated = len(generated_text.split())
        
        return generated_text.strip(), tokens_generated
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Sorry, I encountered an error: {str(e)}", 0

# FastAPI app
app = FastAPI(
    title="Trilingual AI API",
    description="Modern API for multilingual conversation in English, Kiswahili, Kikuyu, and Luo",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting Trilingual AI API Server")
    load_model()
    logger.info("✅ API Server ready")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Trilingual AI API Server",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_info["gpu_memory_cached"] = torch.cuda.memory_reserved() / 1024**2  # MB
    
    # Basic CPU memory (approximation)
    import psutil
    process = psutil.Process()
    memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024**2
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        memory_usage=memory_info,
        api_version="2.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    start_time = time.time()
    
    try:
        # Generate conversation ID if not provided
        conv_id = request.conversation_id or str(uuid.uuid4())
        
        # Initialize conversation history
        if conv_id not in conversations:
            conversations[conv_id] = []
        
        # Add user message to history
        conversations[conv_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Detect language
        detected_lang, confidence = detect_language(request.message)
        
        # Prepare context for generation
        context = request.message
        if len(conversations[conv_id]) > 1:
            # Include recent conversation context
            recent_messages = conversations[conv_id][-3:]  # Last 3 messages
            context_parts = []
            for msg in recent_messages[:-1]:  # Exclude the current message
                if msg["role"] == "user":
                    context_parts.append(f"User: {msg['content']}")
                else:
                    context_parts.append(f"Assistant: {msg['content']}")
            context_parts.append(f"User: {request.message}")
            context = " ".join(context_parts)
        
        # Generate response
        response_text, tokens_used = generate_response(
            context,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        # Add assistant response to history
        conversations[conv_id].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit conversation history (keep last 20 messages)
        if len(conversations[conv_id]) > 20:
            conversations[conv_id] = conversations[conv_id][-20:]
        
        processing_time = time.time() - start_time
        
        # Model info
        model_info = {
            "model_type": "SimpleLLM",
            "parameters": sum(p.numel() for p in model.parameters()) if model else 0,
            "device": str(device),
            "vocabulary_size": tokenizer.vocab_size if tokenizer else 0
        }
        
        logger.info(f"Chat request processed in {processing_time:.2f}s, {tokens_used} tokens generated")
        
        return ChatResponse(
            response=response_text,
            conversation_id=conv_id,
            language_detected=detected_lang,
            confidence=confidence,
            tokens_generated=tokens_used,
            processing_time=processing_time,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id],
        "message_count": len(conversations[conversation_id])
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    return {"message": "Conversation deleted successfully"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
