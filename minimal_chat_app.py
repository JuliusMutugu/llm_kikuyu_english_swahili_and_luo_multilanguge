"""
Minimal Multilingual Chat Interface
Clean, simple, and focused on conversation
"""

import streamlit as st
import requests
import datetime
import json
from typing import Dict, List

# Auto-detect API port
def find_api_port():
    """Find the port where the API is running"""
    import socket
    
    # Try common API ports
    ports_to_try = [8001, 8000, 8002, 8003, 8004]
    
    for port in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    # Port is open, try to verify it's our API
                    try:
                        response = requests.get(f"http://localhost:{port}/", timeout=2)
                        if response.status_code == 200:
                            return port
                    except:
                        continue
        except:
            continue
    
    # Default fallback
    return 8001

API_PORT = find_api_port()
API_BASE_URL = f"http://localhost:{API_PORT}"

# Page config
st.set_page_config(
    page_title="Multilingual Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        margin-right: auto;
    }
    
    .language-selector {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 10px 20px;
    }
    
    .stButton > button {
        border-radius: 25px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Language configurations
LANGUAGES = {
    "english": {"name": "English", "flag": "🇺🇸"},
    "kiswahili": {"name": "Kiswahili", "flag": "🇰🇪"},
    "kikuyu": {"name": "Kikuyu", "flag": "🇰🇪"},
    "luo": {"name": "Luo", "flag": "🇰🇪"}
}

def init_session_state():
    """Initialize session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None

def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def send_message(message: str, language: str) -> Dict:
    """Send message to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "language": language,
                "conversation_id": st.session_state.conversation_id,
                "max_length": 100,
                "temperature": 0.7,
                "model_preference": "best"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

def main():
    """Main app function"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💬 Multilingual Chat Assistant</h1>
        <p>Chat in English, Kiswahili, Kikuyu, or Luo</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API status
    api_online = check_api_status()
    
    if not api_online:
        st.error("🔴 API Server is offline. Please start the server first.")
        st.code("python multi_model_api.py")
        return
    
    # Language selector (minimal)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        selected_language = st.selectbox(
            "🌍 Choose Language:",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: f"{LANGUAGES[x]['flag']} {LANGUAGES[x]['name']}",
            index=0
        )
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                    <br><small>{message.get('timestamp', '')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong> {message['content']}
                    <br><small>{message.get('timestamp', '')} • {message.get('model_used', 'unknown')} • {message.get('language_detected', 'auto')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    
    # Message input
    col_input, col_send = st.columns([4, 1])
    
    with col_input:
        user_input = st.text_input(
            "💭 Type your message:",
            placeholder=f"Type in {LANGUAGES[selected_language]['name']}...",
            key="message_input"
        )
    
    with col_send:
        send_button = st.button("📤 Send", type="primary")
    
    # Handle message sending
    if (send_button or user_input) and user_input.strip():
        # Add user message
        timestamp = datetime.datetime.now().strftime("%H:%M")
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        }
        st.session_state.messages.append(user_message)
        
        # Send to API and get response
        with st.spinner("🤔 Thinking..."):
            response = send_message(user_input, selected_language)
            
            if "error" in response:
                st.error(f"❌ {response['error']}")
            else:
                # Add assistant response
                assistant_message = {
                    "role": "assistant",
                    "content": response.get("response", "No response"),
                    "timestamp": datetime.datetime.now().strftime("%H:%M"),
                    "model_used": response.get("model_used", "unknown"),
                    "language_detected": response.get("language_detected", "auto")
                }
                st.session_state.messages.append(assistant_message)
                
                # Update conversation ID
                if response.get("conversation_id"):
                    st.session_state.conversation_id = response["conversation_id"]
        
        # Clear input and rerun
        st.session_state.message_input = ""
        st.rerun()
    
    # Simple controls at bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
    
    with col2:
        if st.session_state.messages:
            # Export chat
            export_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "total_messages": len(st.session_state.messages)
            }
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 Export Chat",
                data=json_str,
                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        # Show simple stats
        if st.session_state.messages:
            total_messages = len(st.session_state.messages)
            user_messages = sum(1 for m in st.session_state.messages if m['role'] == 'user')
            st.info(f"📊 {total_messages} messages ({user_messages} from you)")

if __name__ == "__main__":
    main()
