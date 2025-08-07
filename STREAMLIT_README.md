# 🧠 Trilingual AI Assistant - Modern Streamlit Interface

A beautiful, modern web interface for conversational AI supporting **English**, **Kiswahili**, **Kikuyu**, and **Luo**.

![Interface Preview](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Languages](https://img.shields.io/badge/Languages-4-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Ready-success?style=for-the-badge)

## ✨ Features

### 🎨 Modern Interface
- **Beautiful Design**: Clean, modern UI with glassmorphism effects
- **Dark Theme**: Easy on the eyes with professional styling
- **Responsive**: Works perfectly on desktop, tablet, and mobile
- **Real-time**: Instant responses with typing indicators

### 🌍 Multilingual Support
- **Auto-detection**: Automatically detects input language
- **4 Languages**: English, Kiswahili, Kikuyu, and Luo
- **Smart Responses**: Context-aware replies in the appropriate language
- **Cultural Context**: Understanding of cultural nuances

### 🚀 Advanced Features
- **Session Management**: Maintains conversation context
- **Export Chat**: Download conversation history as JSON
- **Model Settings**: Adjust temperature, response length, and creativity
- **Real-time Stats**: Token usage and performance metrics
- **Quick Examples**: Pre-built conversation starters

## 🚀 Quick Start

### Method 1: Simple Launcher (Recommended)
```bash
python launcher.py
```

### Method 2: Manual Start
1. **Start API Server**:
   ```bash
   python api_server_modern.py
   ```

2. **Start Streamlit Interface** (in new terminal):
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open Browser**: Go to http://localhost:8501

### Method 3: Batch File (Windows)
```bash
start_streamlit.bat
```

## 🎯 Interface Overview

### Main Components

#### 🏠 Header
- **App Title**: Trilingual AI Assistant branding
- **Language Badges**: Visual indication of supported languages
- **Status Indicators**: Real-time API connectivity status

#### 🗨️ Chat Area
- **Message Bubbles**: User and AI messages with distinct styling
- **Metadata**: Language detection, confidence scores, token count
- **Welcome Screen**: Helpful introduction for new users
- **Export Function**: Download conversation history

#### ⚙️ Sidebar Controls
- **Language Selection**: Choose preferred language or auto-detect
- **Model Settings**: 
  - Temperature (0.1-1.0): Controls creativity/randomness
  - Response Length (50-200): Maximum tokens in response
- **Session Statistics**: Live metrics and usage tracking
- **Quick Examples**: Ready-to-use conversation starters
- **Actions**: Clear chat, export history

### 🎨 Design Features

#### Modern Styling
- **Inter Font**: Professional typography
- **Gradient Backgrounds**: Beautiful color transitions
- **Box Shadows**: Depth and dimension
- **Smooth Animations**: Polished user experience

#### Responsive Design
- **Mobile-First**: Works great on all devices
- **Flexible Layout**: Adapts to different screen sizes
- **Touch-Friendly**: Large buttons and intuitive gestures

## 🛠️ Configuration

### Environment Variables
```bash
# API Configuration
STREAMLIT_SERVER_PORT=8501
API_SERVER_URL=http://localhost:8000

# Model Settings
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_LENGTH=100
```

### Streamlit Config
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
headless = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#0f0f23"
secondaryBackgroundColor = "#1a1a2e"
textColor = "#ffffff"
```

## 🔧 Advanced Usage

### Custom Styling
The interface supports custom CSS modifications in `streamlit_app.py`:
```python
# Modify colors
--primary-color: #667eea;
--secondary-color: #764ba2;

# Adjust typography
font-family: 'Inter', sans-serif;
```

### API Integration
The interface communicates with the API server via REST endpoints:
- `POST /chat`: Send messages and receive responses
- `GET /health`: Check API server status
- `GET /conversations/{id}`: Retrieve conversation history

### Performance Optimization
- **Caching**: API status cached for 30 seconds
- **Session State**: Efficient conversation management
- **Lazy Loading**: Components load as needed

## 📱 Mobile Experience

The interface is fully optimized for mobile devices:
- **Touch Interactions**: Large, touch-friendly buttons
- **Responsive Layout**: Adapts to small screens
- **Fast Loading**: Optimized for mobile networks
- **Offline Indicators**: Clear status when API unavailable

## 🎭 Language Examples

### English
- "Hello, how can you help me today?"
- "Tell me about your capabilities"
- "What languages do you speak?"

### Kiswahili
- "Habari yako? Unaweza kunisaidia?"
- "Niambie kuhusu uwezo wako"
- "Unaongea lugha gani?"

### Kikuyu
- "Wĩ atĩa? Ũngĩndeithagia?"
- "Njĩra cia gũthoma ciũgano"
- "Wĩ mũtaare wa ciũgano?"

### Luo
- "Inadi? Inyalo konya nadi?"
- "Nyisa kuom tekoni magi"
- "Iwacho dhok mage?"

## 🔍 Troubleshooting

### Common Issues

#### API Server Not Starting
```bash
# Check if port 8000 is in use
netstat -an | findstr :8000

# Kill existing process (Windows)
taskkill /F /PID <process_id>
```

#### Streamlit Connection Error
1. Verify API server is running on port 8000
2. Check firewall settings
3. Ensure all dependencies are installed

#### Model Loading Issues
1. Check if model files exist in `checkpoints/`
2. Verify PyTorch installation
3. Check available memory

### Performance Tips
- **Use GPU**: If available, models will automatically use CUDA
- **Adjust Settings**: Lower temperature for faster, more consistent responses
- **Clear History**: Regular conversation cleanup improves performance

## 📊 Monitoring

### Built-in Metrics
- **Message Count**: Total messages in session
- **Token Usage**: Cumulative tokens generated
- **Response Time**: Average API response time
- **Language Distribution**: Usage by language

### Logging
- **API Logs**: `api_server.log`
- **Streamlit Logs**: Console output
- **Error Tracking**: Automatic error reporting

## 🔮 Future Enhancements

### Planned Features
- **Voice Input**: Speech-to-text integration
- **Voice Output**: Text-to-speech responses
- **Theme Switcher**: Light/dark mode toggle
- **Advanced Settings**: More model parameters
- **Chat History**: Persistent conversation storage
- **User Profiles**: Personalized experience

### Community Features
- **Share Conversations**: Export and share interesting chats
- **Language Learning**: Built-in learning tools
- **Cultural Insights**: Educational content about languages

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd trilingual-ai

# Install dependencies
pip install -r requirements.txt

# Run development server
python launcher.py
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Streamlit Team**: For the amazing framework
- **Language Communities**: For cultural insights and feedback
- **Contributors**: Everyone who helped improve this project

---

**Made with ❤️ for multilingual communities**

🌐 [Website](https://your-website.com) | 📧 [Contact](mailto:your-email@domain.com) | 🐛 [Issues](https://github.com/your-repo/issues)
