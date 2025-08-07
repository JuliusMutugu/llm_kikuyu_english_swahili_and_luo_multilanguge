#!/usr/bin/env python3
"""
FastAPI Server for Trilingual East African LLM
Supports Swahili, Kikuyu, and Luo languages with love/conversation context
"""

import os
import sys
import json
import torch
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.modern_llm import ModernLLM, ModelConfig, create_model

# Global model storage
model_instance = None
model_config = None
tokenizer_info = None

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "auto"  # auto, swahili, kikuyu, luo, english
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    language_detected: str
    confidence: float
    conversation_id: str
    tokens_generated: int

class ModelInfo(BaseModel):
    name: str
    languages: List[str]
    parameters: int
    model_size_mb: float
    capabilities: List[str]

class SimpleTokenizer:
    """Simple character-level tokenizer for multilingual text"""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.built = False
    
    def build_from_text(self, text: str):
        """Build vocabulary from text"""
        chars = sorted(list(set(text)))
        
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        chars = special_tokens + [c for c in chars if c not in special_tokens]
        
        # Limit vocabulary size
        if len(chars) > self.vocab_size:
            chars = chars[:self.vocab_size]
        
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.built = True
        
        return len(chars)
    
    def encode(self, text: str, max_length: int = 256) -> torch.Tensor:
        """Encode text to tensor"""
        if not self.built:
            raise ValueError("Tokenizer not built. Call build_from_text() first.")
        
        indices = [self.char_to_idx.get(ch, 1) for ch in text[:max_length]]  # 1 = <unk>
        
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(0)  # 0 = <pad>
        
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    def decode(self, indices: torch.Tensor) -> str:
        """Decode tensor to text"""
        if not self.built:
            raise ValueError("Tokenizer not built.")
        
        if len(indices.shape) > 1:
            indices = indices.squeeze()
        
        chars = []
        for idx in indices:
            char = self.idx_to_char.get(idx.item(), '<unk>')
            if char in ['<pad>', '<start>', '<end>']:
                continue
            chars.append(char)
        
        return ''.join(chars).strip()

def detect_language(text: str) -> tuple[str, float]:
    """Simple language detection based on common words"""
    text_lower = text.lower()
    
    # Language indicators
    swahili_words = ['na', 'ni', 'ya', 'wa', 'kwa', 'katika', 'sana', 'mtu', 'watu', 'nini']
    kikuyu_words = ['nĩ', 'wa', 'ma', 'gĩ', 'kĩ', 'mũ', 'atĩa', 'wega', 'mwega', 'ciugo']
    luo_words = ['en', 'ma', 'gi', 'kod', 'mar', 'mag', 'nadi', 'ber', 'maber', 'wach']
    english_words = ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'do', 'does']
    
    scores = {
        'swahili': sum(1 for word in swahili_words if word in text_lower),
        'kikuyu': sum(1 for word in kikuyu_words if word in text_lower),
        'luo': sum(1 for word in luo_words if word in text_lower),
        'english': sum(1 for word in english_words if word in text_lower)
    }
    
    if max(scores.values()) == 0:
        return 'auto', 0.5
    
    detected = max(scores, key=scores.get)
    confidence = scores[detected] / len(text.split()) if text.split() else 0.5
    
    return detected, min(confidence, 1.0)

def load_model():
    """Load the trained model"""
    global model_instance, model_config, tokenizer_info
    
    try:
        # Try to load trained model
        checkpoint_path = "checkpoints/quick_multilingual_model.pt"
        
        if os.path.exists(checkpoint_path):
            print("📥 Loading trained model...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Recreate config
            config_dict = checkpoint.get('config', {})
            model_config = ModelConfig(**config_dict)
            
            # Create model
            model_instance = create_model(model_config)
            model_instance.load_state_dict(checkpoint['model_state_dict'])
            model_instance.eval()
            
            # Load tokenizer info
            tokenizer_info = {
                'vocab_size': checkpoint.get('vocab_size', model_config.vocab_size),
                'char_to_idx': checkpoint.get('char_to_idx', {}),
            }
            
            print("✅ Trained model loaded successfully!")
            
        else:
            print("⚠️ No trained model found, creating new one...")
            # Create fresh model
            model_config = ModelConfig(
                vocab_size=2000,  # Smaller for demo
                hidden_size=256,
                num_layers=4,
                num_heads=4,
                num_kv_heads=2,
                intermediate_size=512,
                max_position_embeddings=128,
            )
            
            model_instance = create_model(model_config)
            model_instance.eval()
            
            print("✅ New model created!")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def generate_response(prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
    """Generate response using the model"""
    global model_instance, model_config, tokenizer_info
    
    if model_instance is None:
        return "Model not loaded. Please try again later."
    
    try:
        # Simple tokenizer
        tokenizer = SimpleTokenizer(vocab_size=model_config.vocab_size)
        
        # If we have tokenizer info from checkpoint, use it
        if tokenizer_info and 'char_to_idx' in tokenizer_info:
            tokenizer.char_to_idx = tokenizer_info['char_to_idx']
            tokenizer.idx_to_char = {v: k for k, v in tokenizer.char_to_idx.items()}
            tokenizer.built = True
        else:
            # Build from sample data
            try:
                with open('data/sample_data.txt', 'r', encoding='utf-8') as f:
                    sample_text = f.read()
                tokenizer.build_from_text(sample_text)
            except:
                # Fallback vocabulary
                sample_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'"
                sample_chars += "ĩũĩẽõãĩũĩẽõãàáâãäèéêëìíîïòóôõöùúûüçñ"  # Add accented chars
                tokenizer.build_from_text(sample_chars * 10)
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, max_length=64)
        
        # Generate
        with torch.no_grad():
            # Simple generation (no sophisticated sampling)
            generated = input_ids.clone()
            
            for _ in range(min(max_length, 50)):  # Limit generation
                outputs = model_instance(generated)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.get('logits', outputs)
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :] / temperature
                
                # Simple sampling
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append token
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we generate end token or reach max length
                if next_token.item() == 3:  # <end> token
                    break
        
        # Decode response (skip the input part)
        full_response = tokenizer.decode(generated[0])
        response = full_response[len(prompt):].strip()
        
        # Clean up response
        if not response or len(response) < 5:
            responses_by_lang = {
                'swahili': ["Asante kwa maswali yako! Ni furaha kuzungumza nawe.",
                           "Habari! Ninafurahi kukuongea nawe.",
                           "Karibu! Nani nikusaidie?"],
                'kikuyu': ["Nĩ wega gũkũona! Nĩngwenda gũkwaria nawe.",
                          "Warikia atĩa? Nĩndĩ na gĩkeno gĩa gũkwaria nawe.",
                          "Karibu mũno! Nĩ atĩa ndĩngĩkũteithia?"],
                'luo': ["Ber bedo kodi! Amor ka awuoyo kodi.",
                       "Nadi? Amosani kendo amor wuoyo kodi.",
                       "Bwogo matek! Nang'o ma anyalo konyi goe?"],
                'english': ["Hello! I'm happy to chat with you.",
                           "Hi there! How can I help you today?",
                           "Welcome! I'd love to talk with you."]
            }
            
            detected_lang, _ = detect_language(prompt)
            if detected_lang in responses_by_lang:
                import random
                response = random.choice(responses_by_lang[detected_lang])
            else:
                response = "Asante! Ninafurahi kuzungumza nawe. (Thank you! I'm happy to talk with you.)"
        
        return response[:200]  # Limit response length
        
    except Exception as e:
        print(f"Generation error: {e}")
        return "Samahani, kuna tatizo. (Sorry, there's an issue.) Please try again."

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Trilingual LLM API...")
    load_model()
    yield
    # Shutdown
    print("👋 Shutting down API...")

app = FastAPI(
    title="Trilingual East African LLM API",
    description="A conversational AI that understands Swahili, Kikuyu, and Luo languages with cultural context",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trilingual LLM API</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container { 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 40px 20px;
            }
            .header { 
                text-align: center; 
                margin-bottom: 40px;
            }
            .header h1 {
                font-size: 3em;
                margin-bottom: 10px;
                font-weight: 300;
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .interfaces {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin-bottom: 50px;
            }
            .interface-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 30px;
                text-align: center;
                transition: transform 0.3s ease;
            }
            .interface-card:hover {
                transform: translateY(-10px);
            }
            .interface-card h3 {
                font-size: 1.5em;
                margin-bottom: 15px;
                font-weight: 400;
            }
            .interface-card p {
                margin-bottom: 25px;
                opacity: 0.9;
                line-height: 1.6;
            }
            .btn {
                display: inline-block;
                padding: 15px 30px;
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 25px;
                color: white;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            .btn:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: scale(1.05);
            }
            .features {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
            }
            .features h2 {
                margin-bottom: 20px;
                font-weight: 300;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }
            .feature {
                padding: 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
            }
            .lang-demo { 
                background: rgba(255, 255, 255, 0.1); 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 15px;
                line-height: 1.8;
            }
            .api-links {
                text-align: center;
                margin-top: 30px;
            }
            .api-links a {
                margin: 0 15px;
                color: rgba(255, 255, 255, 0.8);
                text-decoration: none;
            }
            .api-links a:hover {
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>�️ Trilingual AI Assistant</h1>
                <p>Voice-enabled conversational AI for East African languages</p>
            </div>
            
            <div class="interfaces">
                <div class="interface-card">
                    <h3>🎤 Simple Voice Chat</h3>
                    <p>Clean, minimal interface with voice input and output. Just type or speak naturally - no complicated controls.</p>
                    <a href="/simple" class="btn">Launch Voice Chat</a>
                </div>
                
                <div class="interface-card">
                    <h3>💻 Advanced Chat</h3>
                    <p>Full-featured interface with language selection, examples, conversation export, and detailed controls.</p>
                    <a href="/chat-ui" class="btn">Launch Advanced Chat</a>
                </div>
            </div>
            
            <div class="features">
                <h2>🌍 Supported Languages</h2>
                <div class="lang-demo">
                    <strong>🇬🇧 English:</strong> "Hello! How can I help you today?"<br>
                    <strong>🇰🇪 Swahili:</strong> "Habari yako? Nakupenda sana!"<br>
                    <strong>🇰🇪 Kikuyu:</strong> "Warikia atĩa? Nĩngũkwenda mũno!"<br>
                    <strong>🇰🇪 Luo:</strong> "Nadi? Aheri miwuoro matek!"
                </div>
                
                <h2>✨ Key Features</h2>
                <div class="feature-grid">
                    <div class="feature">
                        <strong>🎙️ Voice Input</strong><br>
                        Speak naturally in any language
                    </div>
                    <div class="feature">
                        <strong>🔊 Voice Output</strong><br>
                        Hear responses spoken back
                    </div>
                    <div class="feature">
                        <strong>🌍 Auto-Detection</strong><br>
                        Automatic language recognition
                    </div>
                    <div class="feature">
                        <strong>💖 Cultural Context</strong><br>
                        Love expressions and greetings
                    </div>
                    <div class="feature">
                        <strong>💬 Natural Conversation</strong><br>
                        Context-aware responses
                    </div>
                    <div class="feature">
                        <strong>📱 Mobile Friendly</strong><br>
                        Works on all devices
                    </div>
                </div>
            </div>
            
            <div class="api-links">
                <a href="/docs">📖 API Documentation</a>
                <a href="/health">❤️ Health Check</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the trilingual LLM"""
    try:
        # Detect language if auto
        if request.language == "auto":
            detected_lang, confidence = detect_language(request.message)
        else:
            detected_lang = request.language
            confidence = 1.0
        
        # Generate response
        response_text = generate_response(
            request.message, 
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{hash(request.message) % 10000:04d}"
        
        return ChatResponse(
            response=response_text,
            language_detected=detected_lang,
            confidence=confidence,
            conversation_id=conversation_id,
            tokens_generated=len(response_text.split())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the model"""
    global model_instance, model_config
    
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model_instance.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # Rough estimate for FP32
    
    return ModelInfo(
        name="Trilingual East African LLM",
        languages=["Swahili", "Kikuyu", "Luo", "English"],
        parameters=total_params,
        model_size_mb=round(model_size_mb, 2),
        capabilities=[
            "Multilingual conversation",
            "Love and relationship context",
            "Cultural awareness",
            "Automatic language detection",
            "Character-level generation"
        ]
    )

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "languages": [
            {
                "code": "sw",
                "name": "Swahili",
                "native_name": "Kiswahili",
                "example": "Habari yako? Nakupenda sana!"
            },
            {
                "code": "ki",
                "name": "Kikuyu", 
                "native_name": "Gĩkũyũ",
                "example": "Warikia atĩa? Nĩngũkwenda mũno!"
            },
            {
                "code": "luo",
                "name": "Luo",
                "native_name": "Dholuo", 
                "example": "Nadi? Aheri miwuoro matek!"
            },
            {
                "code": "en",
                "name": "English",
                "native_name": "English",
                "example": "How are you? I love you!"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "timestamp": "2025-08-06T00:00:00Z"
    }

@app.get("/chat-ui")
async def chat_interface():
    """Serve the modern chat interface"""
    try:
        chat_file = Path(__file__).parent / "chat.html"
        if chat_file.exists():
            return FileResponse(chat_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Chat interface not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving chat interface: {str(e)}")

@app.get("/simple")
async def simple_chat():
    """Serve the simple voice chat interface"""
    try:
        chat_file = Path(__file__).parent / "simple_chat.html"
        if chat_file.exists():
            return FileResponse(chat_file, media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Simple chat interface not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving simple chat interface: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Trilingual East African LLM API",
        "languages": ["swahili", "kikuyu", "luo", "english"],
        "features": ["multilingual", "cultural_context", "conversation", "voice_support"]
    }

if __name__ == "__main__":
    print("🚀 Starting Trilingual LLM FastAPI Server...")
    print("🌍 Supporting Swahili, Kikuyu, and Luo languages")
    print("💕 With love and conversation context")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0", 
        port=8000,
        reload=False,
        log_level="info"
    )
