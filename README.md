# 🎙️ Voice AI Assistant - Trilingual East African LLM

A **simplified, voice-enabled** conversational AI that understands **English**, **Swahili**, **Kikuyu**, and **Luo** with natural speech communication.

## ✨ Key Features

### 🎤 **Voice Input & Output**
- **Speak naturally** - just talk in any language
- **Automatic speech recognition** in all supported languages  
- **Text-to-speech responses** with natural voice
- **Clean, minimal interface** - no complex controls

### 🌍 **Multilingual Support**
- **English**: "Hello! How can you help me today?"
- **Swahili**: "Habari! Unaweza kunisaidia?"
- **Kikuyu**: "Wĩ mwega! Ũngĩndeithagia atĩa?"
- **Luo**: "Inadi! Inyalo konya nadi?"

### 💖 **Cultural Context**
- Love expressions: "Nakupenda sana" → "Aheri miwuoro matek"
- Traditional greetings and natural conversations
- East African cultural awareness

## 🚀 **Ultra-Simple Setup**

### 1. **Start the Server**
```bash
# Windows - Double click or run:
start_voice_ai.bat

# Or manually:
python api_server.py
```

### 2. **Open Voice Chat**
- **Simple Voice Chat**: http://localhost:8000/simple ⭐ **RECOMMENDED**
- **Advanced Chat**: http://localhost:8000/chat-ui
- **Home Page**: http://localhost:8000

## 🎯 **How to Use - Super Simple!**

### **Option 1: Voice (Recommended)**
1. Click the **🎤 microphone button**
2. **Speak naturally** in any language
3. **Listen to the AI response**

### **Option 2: Text**
1. **Type your message** in the text box
2. Press **Enter** or click **Send**
3. **Hear the spoken response**

### **That's it!** No complex settings, no language selection needed.

## 💬 **Try These Examples**

Just speak or type:

**English:**
- "Hello, how are you?"
- "Tell me a story"
- "I love you"

**Swahili:**
- "Habari yako?"
- "Nakupenda sana"
- "Niambie hadithi"

**Kikuyu:**
- "Wĩ atĩa?"
- "Nĩngũkwenda mũno"
- "Njĩra ya rũgano"

**Luo:**
- "Inadi?"
- "Aheri miwuoro matek" 
- "Nyisa sigana"

## 🎨 **Beautiful, Simple Interface**

- **Glass-morphism design** with blurred backgrounds
- **Smooth animations** and typing indicators
- **Mobile-responsive** - works on phones and tablets
- **Voice status indicators** - see when AI is listening/speaking
- **Auto-scrolling chat** - focus on conversation
- **Error handling** - graceful connection management

## 🔧 **Technical Features**

### **Voice Technology**
- **WebRTC Speech Recognition** - works in modern browsers
- **Speech Synthesis API** - natural voice output
- **Auto-language detection** - no manual selection needed
- **Voice activity detection** - smart listening

### **AI Capabilities** 
- **Context-aware responses** maintaining conversation flow
- **Cultural sensitivity** with appropriate expressions
- **Real-time language detection** with confidence scoring
- **Conversation memory** across multiple exchanges

### **Modern Architecture**
- **FastAPI backend** with automatic documentation
- **PyTorch 2.1+** with latest optimizations
- **RMSNorm, SwiGLU, RoPE** - cutting-edge transformer tech
- **Character-level tokenization** for multilingual support
- **CORS enabled** - works from any domain

## 📁 **Project Structure**

```
llm/
├── simple_chat.html      # 🌟 Main voice interface (RECOMMENDED)
├── chat.html            # Advanced chat with full controls  
├── api_server.py        # FastAPI server with voice support
├── start_voice_ai.bat   # Easy Windows startup
├── modern_llm.py        # Core AI model
├── data/
│   └── sample_data.txt  # Trilingual training data
└── configs/
    └── config.yaml      # Model configuration
```

## 🌟 **Why This AI is Special**

### **Unprecedented Simplicity**
- **Just speak or type** - no complexity
- **Auto-everything** - language detection, voice output
- **One-click startup** - works immediately  
- **Zero configuration** - smart defaults

### **Cultural Intelligence**
- **First trilingual Kenyan AI** with deep cultural context
- **Love expressions** in all languages
- **Traditional greetings** and conversational patterns
- **Code-switching support** - mix languages naturally

### **Technical Excellence**
- **State-of-the-art architecture** - 2025 standards
- **Optimized for laptops** - runs efficiently
- **Web-based** - no app installation needed
- **Voice-first design** - natural human interaction

## 🎉 **Quick Demo**

1. **Double-click** `start_voice_ai.bat`
2. **Open** http://localhost:8000/simple  
3. **Click** the microphone 🎤
4. **Say** "Habari yako?" (How are you in Swahili)
5. **Listen** to the AI respond in Swahili with voice!

## 🔮 **Coming Soon**

- **Offline voice processing** - no internet needed
- **Multiple voice personalities** - choose your AI's voice
- **Conversation history** - save and replay chats
- **Mobile app** - native iOS/Android experience
- **Real-time translation** - speak in one language, hear in another

---

**🚀 Experience the future of multilingual AI conversation!** 

Built with ❤️ for natural, voice-first interaction in East African languages.

**Just speak. The AI understands.** 🎙️✨
- **RetNet**: Alternative to Transformer with better scaling

## Project Structure

```
llm/
├── src/
│   ├── models/          # Model architectures
│   ├── training/        # Training loops and optimization
│   ├── data/           # Data processing and tokenization
│   ├── inference/      # Inference engines and optimization
│   └── utils/          # Utilities and helpers
├── notebooks/          # Jupyter notebooks for experimentation
├── configs/           # Configuration files
├── data/             # Training data
└── experiments/      # Experiment tracking
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Configure your environment: `cp configs/config.yaml.example configs/config.yaml`
3. Explore the notebooks in `notebooks/` for hands-on learning
4. Run training: `python src/training/train.py --config configs/small_model.yaml`

## Features

- ✅ Modern Transformer architecture with latest optimizations
- ✅ Mixture of Experts implementation
- ✅ Parameter-efficient fine-tuning (LoRA)
- ✅ Instruction tuning capabilities
- ✅ Efficient inference with KV-caching
- ✅ Multi-GPU training support
- ✅ Comprehensive evaluation suite
- ✅ Interactive chat interface

## Recent Papers Implemented

- "Attention Is All You Need" (Transformer baseline)
- "Switch Transformer: Scaling to Trillion Parameter Models"
- "LLaMA: Open and Efficient Foundation Language Models"
- "Constitutional AI: Harmlessness from AI Feedback"
- "LoRA: Low-Rank Adaptation of Large Language Models"
- "RoFormer: Enhanced Transformer with Rotary Position Embedding"
