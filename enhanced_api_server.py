#!/usr/bin/env python3
"""
Enhanced API Server for Trilingual LLM
Uses improved model with better performance
"""

import os
import sys
from pathlib import Path
import json
import logging
from typing import Optional, Dict, Any

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("FastAPI not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "jinja2", "python-multipart"])
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel
    import uvicorn

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
except ImportError:
    print("PyTorch not installed. Please install with: pip install torch numpy")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.8

class ChatResponse(BaseModel):
    response: str
    language: str
    confidence: float

class SimpleModel(nn.Module):
    """Simple but effective language model"""
    
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.output(lstm_out)

class TrilingualLLM:
    """Enhanced Trilingual Language Model"""
    
    def __init__(self, model_path: str = "checkpoints/quick_multilingual_model.pt"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = 0
        
        # Language patterns for detection
        self.language_patterns = {
            'english': ['hello', 'good', 'thank', 'how', 'what', 'where', 'when', 'why'],
            'swahili': ['habari', 'asante', 'karibu', 'hujambo', 'nzuri', 'sawa', 'haya'],
            'kikuyu': ['atia', 'ngai', 'wi', 'ni', 'uria', 'kuga', 'guca', 'thii'],
            'luo': ['nadi', 'erokamano', 'amosi', 'ka', 'nene', 'nade', 'miya', 'dhi']
        }
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            if not Path(model_path).exists():
                logger.warning(f"Model not found at {model_path}, using fallback")
                self._create_fallback_model()
                return
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load tokenizer data
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']
            self.vocab_size = checkpoint['vocab_size']
            
            # Create and load model
            self.model = SimpleModel(self.vocab_size).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Vocabulary size: {self.vocab_size}")
            logger.info(f"Device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a fallback model with basic responses"""
        # Create basic character mapping
        basic_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-")
        additional_chars = list("ñáéíóúàèìòùâêîôûäëïöüãõç")  # Common accented characters
        swahili_chars = list("ĩũ")  # Kikuyu specific
        
        all_chars = basic_chars + additional_chars + swahili_chars
        
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for i, char in enumerate(all_chars):
            self.char_to_idx[char] = i + 4
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Create simple model
        self.model = SimpleModel(self.vocab_size).to(self.device)
        self.model.eval()
        
        logger.info("Created fallback model")
    
    def encode_text(self, text: str, max_len: int = 64) -> list:
        """Encode text to token IDs"""
        tokens = [2]  # BOS
        for char in text[:max_len-2]:
            tokens.append(self.char_to_idx.get(char, 1))  # UNK if not found
        tokens.append(3)  # EOS
        while len(tokens) < max_len:
            tokens.append(0)  # PAD
        return tokens
    
    def decode_tokens(self, tokens: list) -> str:
        """Decode token IDs to text"""
        chars = []
        for token in tokens:
            if token in [0, 2, 3]:  # Skip special tokens
                continue
            char = self.idx_to_char.get(token, '<UNK>')
            if char != '<UNK>':
                chars.append(char)
        return ''.join(chars).strip()
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        text_lower = text.lower()
        
        # Count matches for each language
        language_scores = {}
        for lang, patterns in self.language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            language_scores[lang] = score
        
        # Return language with highest score, default to English
        if max(language_scores.values()) > 0:
            return max(language_scores, key=language_scores.get)
        return 'english'
    
    def get_cultural_response(self, message: str, detected_lang: str) -> Optional[str]:
        """Get culturally appropriate response based on message"""
        message_lower = message.lower()
        
        # Greeting responses
        if any(word in message_lower for word in ['hello', 'hi', 'habari', 'hujambo', 'atia', 'nadi']):
            responses = {
                'english': "Hello! How are you doing today?",
                'swahili': "Habari yako! Hujambo?",
                'kikuyu': "Wĩ atĩa? Ũrĩ nĩarĩa?",
                'luo': "Inadi? To idhi nadi?"
            }
            return responses.get(detected_lang, responses['english'])
        
        # Love expressions
        if any(word in message_lower for word in ['love', 'nakupenda', 'kenda', 'aheri']):
            responses = {
                'english': "That's beautiful! Love is wonderful.",
                'swahili': "Hilo ni zuri sana! Upendo ni kitu kizuri.",
                'kikuyu': "Nĩ gũkena kũega! Wendani nĩ kĩega.",
                'luo': "Mano ber ahinya! Hera en gima ber."
            }
            return responses.get(detected_lang, responses['english'])
        
        # Thanks responses
        if any(word in message_lower for word in ['thank', 'asante', 'ngaatho', 'erokamano']):
            responses = {
                'english': "You're very welcome! Happy to help.",
                'swahili': "Karibu sana! Nifurahi kukusaidia.",
                'kikuyu': "Ũrĩ mũgeni! Nĩngenete gũkũteithia.",
                'luo': "Kwa kamano! Amor ka konyou."
            }
            return responses.get(detected_lang, responses['english'])
        
        # Family questions
        if any(word in message_lower for word in ['family', 'familia', 'nyumba', 'joodu']):
            responses = {
                'english': "Family is very important. How is your family?",
                'swahili': "Familia ni muhimu sana. Familia yako hali gani?",
                'kikuyu': "Nyũmba nĩ kĩene mũno. Nyũmba yaku ĩrĩ atĩa?",
                'luo': "Joodu en gima maduong'. Joodu to nade?"
            }
            return responses.get(detected_lang, responses['english'])
        
        return None
    
    def generate_response(self, message: str, max_length: int = 100, temperature: float = 0.8) -> tuple:
        """Generate response to input message"""
        try:
            # Detect language
            detected_lang = self.detect_language(message)
            
            # Try cultural response first
            cultural_response = self.get_cultural_response(message, detected_lang)
            if cultural_response:
                return cultural_response, detected_lang, 0.9
            
            # Use model generation
            if self.model is None:
                return f"Hello! I understand you said: {message}", detected_lang, 0.5
            
            # Encode input
            input_tokens = self.encode_text(message, max_len=32)
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                generated = input_tensor.clone()
                
                # Generate response
                for _ in range(min(max_length, 50)):
                    if generated.size(1) >= 80:  # Prevent too long sequences
                        break
                    
                    outputs = self.model(generated)
                    logits = outputs[:, -1, :] / temperature
                    
                    # Top-k sampling for better quality
                    top_k = 20
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        probs = F.softmax(top_k_logits, dim=-1)
                        next_token_idx = torch.multinomial(probs, 1)
                        next_token = top_k_indices.gather(-1, next_token_idx)
                    else:
                        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # Stop at EOS
                    if next_token.item() == 3:
                        break
            
            # Decode response
            response = self.decode_tokens(generated[0].cpu().tolist())
            
            # Clean up response
            response = response.replace(message, '').strip()
            if not response:
                response = self.get_fallback_response(detected_lang)
            
            return response, detected_lang, 0.7
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            detected_lang = self.detect_language(message)
            return self.get_fallback_response(detected_lang), detected_lang, 0.3
    
    def get_fallback_response(self, language: str) -> str:
        """Get fallback response when generation fails"""
        responses = {
            'english': "I understand you. Could you tell me more?",
            'swahili': "Naelewa. Unaweza kuniambia zaidi?",
            'kikuyu': "Nĩndĩkũmenya. Nĩũngĩnjiira ũngĩ?",
            'luo': "Awinjo. Inyalo nyisa mangeny?"
        }
        return responses.get(language, responses['english'])

# Initialize the model
llm = TrilingualLLM()

# Create FastAPI app
app = FastAPI(
    title="Enhanced Trilingual LLM API",
    description="Improved trilingual language model supporting English, Swahili, Kikuyu, and Luo",
    version="2.0.0"
)

# Mount static files and templates
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

if Path("templates").exists():
    templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat interface"""
    try:
        # Try to serve the enhanced chat interface
        chat_file = Path("simple_chat.html")
        if chat_file.exists():
            with open(chat_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            # Return simple HTML interface
            return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trilingual AI Chat</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                    .chat-container { border: 1px solid #ccc; height: 400px; overflow-y: auto; padding: 20px; margin-bottom: 20px; }
                    .input-container { display: flex; }
                    input { flex: 1; padding: 10px; margin-right: 10px; }
                    button { padding: 10px 20px; }
                    .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
                    .user { background-color: #e3f2fd; text-align: right; }
                    .bot { background-color: #f3e5f5; }
                </style>
            </head>
            <body>
                <h1>Enhanced Trilingual AI Chat</h1>
                <div class="chat-container" id="chat"></div>
                <div class="input-container">
                    <input type="text" id="messageInput" placeholder="Type your message... (English, Swahili, Kikuyu, or Luo)">
                    <button onclick="sendMessage()">Send</button>
                </div>
                <script>
                    async function sendMessage() {
                        const input = document.getElementById('messageInput');
                        const message = input.value.trim();
                        if (!message) return;
                        
                        const chat = document.getElementById('chat');
                        
                        // Add user message
                        chat.innerHTML += `<div class="message user">You: ${message}</div>`;
                        
                        // Clear input
                        input.value = '';
                        
                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({message: message})
                            });
                            
                            const data = await response.json();
                            chat.innerHTML += `<div class="message bot">AI (${data.language}): ${data.response}</div>`;
                        } catch (error) {
                            chat.innerHTML += `<div class="message bot">Error: Could not get response</div>`;
                        }
                        
                        chat.scrollTop = chat.scrollHeight;
                    }
                    
                    document.getElementById('messageInput').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') sendMessage();
                    });
                </script>
            </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving root: {e}")
        return HTMLResponse(content="<h1>Error loading interface</h1>")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for text-based conversation"""
    try:
        response, language, confidence = llm.generate_response(
            request.message,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response,
            language=language,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-chat")
async def voice_chat(request: ChatRequest):
    """Enhanced voice chat endpoint"""
    try:
        response, language, confidence = llm.generate_response(
            request.message,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return JSONResponse({
            "response": response,
            "language": language,
            "confidence": confidence,
            "tts_text": response  # For text-to-speech
        })
    except Exception as e:
        logger.error(f"Error in voice chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": llm.model is not None,
        "device": llm.device,
        "vocab_size": llm.vocab_size
    }

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "languages": ["english", "swahili", "kikuyu", "luo"],
        "patterns": llm.language_patterns
    }

if __name__ == "__main__":
    print("Enhanced Trilingual LLM API Server")
    print("=" * 40)
    print(f"Model device: {llm.device}")
    print(f"Vocabulary size: {llm.vocab_size}")
    print("Starting server on http://localhost:8000")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
