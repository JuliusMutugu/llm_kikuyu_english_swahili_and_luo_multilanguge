"""
Multi-Model API Server with Conversational MoE Integration
Combines multiple models for best conversational experience
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import datetime
import json
import os
from typing import Dict, List, Optional
import logging

# Import our models
try:
    from conversational_moe_model import ConversationalMoESystem
    MOE_AVAILABLE = True
except ImportError:
    print("MoE model not available")
    MOE_AVAILABLE = False

# Import enhanced conversation system
try:
    from smart_conversation import generate_smart_response
    SMART_CONVERSATION_AVAILABLE = True
    print("✅ Smart conversation system loaded")
except ImportError:
    print("Smart conversation system not available")
    SMART_CONVERSATION_AVAILABLE = False

# Import continuous learning system
try:
    from continuous_learning import enhance_response_with_learning, learning_system
    LEARNING_AVAILABLE = True
    print("✅ Continuous learning system loaded")
except ImportError:
    print("Continuous learning system not available")
    LEARNING_AVAILABLE = False

# Import original model components
import sys
sys.path.append('.')

try:
    from quick_train_api import QuickLLM, load_model_for_inference
    ORIGINAL_AVAILABLE = True
except ImportError:
    print("Original model not available")
    ORIGINAL_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Model Conversational API", version="2.0.0")

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    language: str = "auto"
    conversation_id: Optional[str] = None
    max_length: int = 100
    temperature: float = 0.7
    model_preference: str = "best"  # "moe", "original", "best"

class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    language_detected: str
    confidence: float
    tokens_generated: int
    model_used: str
    expert_usage: Optional[List[float]] = None
    processing_time: float

class ModelStatus(BaseModel):
    moe_model_loaded: bool
    original_model_loaded: bool
    total_conversations: int
    uptime: str

class MultiModelSystem:
    """System that manages multiple models for optimal conversation"""
    
    def __init__(self):
        self.moe_system = None
        self.original_model = None
        self.conversations = {}
        self.total_conversations = 0
        self.start_time = datetime.datetime.now()
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        logger.info("Loading conversational models...")
        
        # Load MoE model
        if MOE_AVAILABLE:
            try:
                if os.path.exists("conversational_moe_trained.pt"):
                    self.moe_system = ConversationalMoESystem("conversational_moe_trained.pt")
                    logger.info("✅ Conversational MoE model loaded")
                else:
                    self.moe_system = ConversationalMoESystem()
                    logger.info("✅ Conversational MoE model initialized (untrained)")
            except Exception as e:
                logger.error(f"❌ Failed to load MoE model: {e}")
        
        # Load original model
        if ORIGINAL_AVAILABLE:
            try:
                if os.path.exists("checkpoints/quick_trilingual_model.pt"):
                    model_components = load_model_for_inference("checkpoints/quick_trilingual_model.pt")
                    if model_components:
                        self.original_model = model_components
                        logger.info("✅ Original trilingual model loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load original model: {e}")
        
        logger.info(f"Models loaded: MoE={self.moe_system is not None}, Original={self.original_model is not None}")
    
    def _detect_language(self, text: str) -> tuple:
        """Simple language detection"""
        text_lower = text.lower()
        
        # Kikuyu indicators
        kikuyu_words = ['wĩ', 'atĩa', 'ũrĩ', 'nĩ', 'ũ', 'ĩ', 'gĩkũyũ', 'njĩra']
        if any(word in text_lower for word in kikuyu_words):
            return "kikuyu", 0.8
        
        # Luo indicators  
        luo_words = ['inadi', 'adhi', 'maber', 'gima', 'inyalo', 'konya', 'ere', 'dhok']
        if any(word in text_lower for word in luo_words):
            return "luo", 0.8
            
        # Kiswahili indicators
        swahili_words = ['habari', 'mzuri', 'asante', 'karibu', 'pole', 'haraka', 'lugha', 'sana']
        if any(word in text_lower for word in swahili_words):
            return "kiswahili", 0.8
            
        # Default to English
        return "english", 0.6
    
    def _choose_best_model(self, message: str, language: str, model_preference: str) -> str:
        """Choose the best model for the conversation"""
        if model_preference == "moe" and self.moe_system:
            return "moe"
        elif model_preference == "original" and self.original_model:
            return "original"
        elif model_preference == "best":
            # For multilingual conversations, prefer MoE
            if language in ["kikuyu", "luo", "kiswahili"] and self.moe_system:
                return "moe"
            # For English, use original if available, otherwise MoE
            elif self.moe_system:
                return "moe"
            elif self.original_model:
                return "original"
        
        # Default fallback
        if self.moe_system:
            return "moe"
        elif self.original_model:
            return "original"
        else:
            raise HTTPException(status_code=503, detail="No models available")
    
    def generate_response(
        self, 
        message: str, 
        language: str = "auto",
        conversation_id: str = None,
        max_length: int = 100,
        temperature: float = 0.7,
        model_preference: str = "best"
    ) -> Dict:
        """Generate response using enhanced conversation system"""
        
        start_time = datetime.datetime.now()
        
        # Use smart conversation system as primary method
        if SMART_CONVERSATION_AVAILABLE:
            try:
                smart_result = generate_smart_response(message, conversation_id, language)
                
                # Track conversation
                self.total_conversations += 1
                
                # Enhance response with learning if available
                if LEARNING_AVAILABLE:
                    enhanced_result = enhance_response_with_learning(
                        message,
                        smart_result['response'],
                        smart_result.get('response_language', smart_result['language_detected']),
                        smart_result.get('conversation_id', conversation_id),
                        smart_result['confidence']
                    )
                    
                    # Format response with learning enhancements
                    response_data = {
                        "response": enhanced_result['response'],
                        "conversation_id": smart_result.get('conversation_id', conversation_id),
                        "language_detected": smart_result['language_detected'],
                        "confidence": smart_result['confidence'] + enhanced_result['confidence_boost'],
                        "tokens_generated": smart_result['tokens_generated'],
                        "model_used": "smart_conversation_with_learning",
                        "expert_usage": None,
                        "processing_time": (datetime.datetime.now() - start_time).total_seconds(),
                        "learning_applied": enhanced_result['learning_applied'],
                        "suggestions_used": enhanced_result['suggestions_used'],
                        "web_knowledge_used": enhanced_result['web_knowledge_used']
                    }
                else:
                    # Format response without learning enhancements
                    response_data = {
                        "response": smart_result['response'],
                        "conversation_id": smart_result.get('conversation_id', conversation_id),
                        "language_detected": smart_result.get('response_language', smart_result['language_detected']),
                        "confidence": smart_result['confidence'],
                        "tokens_generated": smart_result['tokens_generated'],
                        "model_used": "smart_conversation",
                        "expert_usage": None,
                        "processing_time": (datetime.datetime.now() - start_time).total_seconds()
                    }
                
                return response_data
                
            except Exception as e:
                logger.error(f"Smart conversation system failed: {e}")
                # Fall through to backup methods
        
        # Fallback to MoE system if available
        if model_preference != "original" and self.moe_system:
            try:
                result = self.moe_system.generate_response(
                    message,
                    max_length=max_length,
                    temperature=temperature,
                    conversation_id=conversation_id
                )
                
                # Enhanced result with better language detection
                detected_lang, confidence = self._detect_language(message)
                result.update({
                    "language_detected": detected_lang,
                    "confidence": confidence,
                    "model_used": "conversational_moe_fallback"
                })
                
                self.total_conversations += 1
                return result
                
            except Exception as e:
                logger.error(f"MoE model failed: {e}")
        
        # Final fallback to simple template responses
        detected_lang, confidence = self._detect_language(message)
        
        # Simple template responses for fallback
        template_responses = {
            "english": [
                "I understand. Could you tell me more about that?",
                "That's interesting. What would you like to know?",
                "I'm here to help. How can I assist you today?",
                "Thank you for sharing that with me."
            ],
            "kiswahili": [
                "Naelewa. Unaweza kuniambia zaidi kuhusu hilo?",
                "Hilo ni la kuvutia. Unataka kujua nini?",
                "Nipo hapa kukusaidia. Naweza kukusaidia vipi leo?",
                "Asante kwa kunishirikisha hilo."
            ],
            "kikuyu": [
                "Nĩndamenya. Ũngĩnjĩra ũhoro ũngĩ ũkoniĩ ũguo?",
                "Ũcio nĩ wa kũgegania. Ũrenda kũmenya atĩa?",
                "Ndĩ gũkũ gũgũteithia. Ingĩgũteithia atĩa ũmũthĩ?",
                "Nĩngũkena nĩ ũndũ wa kũnjĩra ũguo."
            ],
            "luo": [
                "Awinjo. Inyalo nyisa gimoro machielo kuom mano?",
                "Mano konyo chuny. Idwaro ngʼeyo angʼo?",
                "An ka mondo akonyi. Anyalo konyigo kaka nadi kawuono?",
                "Erokamano kuom pogo mano koda."
            ]
        }
        
        import random
        responses = template_responses.get(detected_lang, template_responses["english"])
        response_text = random.choice(responses)
        
        # Generate conversation ID if needed
        if not conversation_id:
            conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total_conversations}"
        
        self.total_conversations += 1
        
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "language_detected": detected_lang,
            "confidence": confidence,
            "tokens_generated": len(response_text.split()),
            "model_used": "template_fallback",
            "expert_usage": None,
            "processing_time": (datetime.datetime.now() - start_time).total_seconds()
        }
# Initialize model system
model_system = MultiModelSystem()

@app.get("/", summary="API Health Check")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Multi-Model Conversational API",
        "version": "2.0.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "models": {
            "moe_available": model_system.moe_system is not None,
            "original_available": model_system.original_model is not None
        }
    }

@app.get("/health", summary="Health Check")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "online",
        "message": "Multi-Model Conversational API",
        "version": "2.0.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "models": {
            "moe_available": model_system.moe_system is not None,
            "original_available": model_system.original_model is not None
        }
    }

@app.post("/chat", response_model=ChatResponse, summary="Generate conversational response")
def chat_endpoint(request: ChatRequest):
    """Generate a conversational response using the best available model"""
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        result = model_system.generate_response(
            message=request.message,
            language=request.language,
            conversation_id=request.conversation_id,
            max_length=request.max_length,
            temperature=request.temperature,
            model_preference=request.model_preference
        )
        
        return ChatResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status", response_model=ModelStatus, summary="Get API status")
def status_endpoint():
    """Get detailed API status"""
    uptime = datetime.datetime.now() - model_system.start_time
    
    return ModelStatus(
        moe_model_loaded=model_system.moe_system is not None,
        original_model_loaded=model_system.original_model is not None,
        total_conversations=model_system.total_conversations,
        uptime=str(uptime)
    )

@app.get("/conversations/{conversation_id}", summary="Get conversation history")
def get_conversation(conversation_id: str):
    """Get conversation history by ID"""
    if conversation_id not in model_system.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": model_system.conversations[conversation_id],
        "total_messages": len(model_system.conversations[conversation_id])
    }

@app.get("/models/test", summary="Test all available models")
def test_models():
    """Test all available models with sample inputs"""
    test_prompts = [
        "Hello, how are you?",
        "Habari yako?",
        "Wĩ atĩa?",
        "Inadi?"
    ]
    
    results = {}
    
    for prompt in test_prompts:
        try:
            result = model_system.generate_response(
                prompt, 
                max_length=30,
                model_preference="best"
            )
            results[prompt] = {
                "response": result["response"],
                "model_used": result["model_used"],
                "language_detected": result["language_detected"]
            }
        except Exception as e:
            results[prompt] = {"error": str(e)}
    
    return {
        "test_results": results,
        "models_status": {
            "moe_available": model_system.moe_system is not None,
            "original_available": model_system.original_model is not None
        }
    }

@app.get("/learning/stats", summary="Get learning system statistics")
def learning_stats():
    """Get comprehensive learning system statistics"""
    if not LEARNING_AVAILABLE:
        return {"error": "Learning system not available"}
    
    try:
        stats = learning_system.get_learning_stats()
        return {
            "learning_system_active": True,
            "statistics": stats,
            "capabilities": {
                "conversation_learning": True,
                "web_learning": True,
                "pattern_recognition": True,
                "knowledge_enhancement": True
            }
        }
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        return {"error": f"Failed to get learning statistics: {str(e)}"}

@app.post("/learning/update", summary="Manually update web knowledge")
def update_web_knowledge(language: str = "english"):
    """Manually trigger web knowledge update"""
    if not LEARNING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Learning system not available")
    
    try:
        count = learning_system.update_web_knowledge(language)
        return {
            "success": True,
            "language": language,
            "knowledge_items_updated": count,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating web knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update knowledge: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Find available port
    def find_available_port(start_port=8000):
        for port in range(start_port, start_port + 10):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    # Test binding to all interfaces (0.0.0.0) not just localhost
                    s.bind(('0.0.0.0', port))
                    return port
                except OSError:
                    continue
        return None  # No port available
    
    available_port = find_available_port(8000)
    
    if available_port is None:
        print("❌ No available ports found between 8000-8009")
        print("💡 Try running: python port_manager.py")
        exit(1)
    
    print("🚀 Starting Multi-Model Conversational API...")
    print(f"   MoE Model: {'✅ Loaded' if model_system.moe_system else '❌ Not loaded'}")
    print(f"   Original Model: {'✅ Loaded' if model_system.original_model else '❌ Not loaded'}")
    print(f"\n🌐 API will be available at: http://localhost:{available_port}")
    print(f"📚 Documentation at: http://localhost:{available_port}/docs")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=available_port)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("💡 Try running: python port_manager.py")
