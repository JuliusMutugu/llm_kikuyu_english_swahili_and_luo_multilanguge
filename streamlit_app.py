#!/usr/bin/env python3
"""
Modern Streamlit Interface for Enhanced Trilingual LLM
Beautiful, intuitive, and feature-rich web interface
"""

import streamlit as st
import requests
import json
import time
import datetime
from pathlib import Path
import base64
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="Trilingual AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "# Trilingual AI Assistant\nPowered by advanced language models supporting English, Kiswahili, Kikuyu, and Luo."
    }
)

# Custom CSS for modern styling
def load_css():
    st.markdown("""
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #43e97b;
        --warning-color: #feca57;
        --error-color: #ff6b6b;
        --text-primary: #2c3e50;
        --text-secondary: #7f8c8d;
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --border-color: #e9ecef;
        --shadow: 0 2px 10px rgba(0,0,0,0.1);
        --border-radius: 12px;
    }
    
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: var(--shadow);
    }
    
    .app-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .language-badges {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .language-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Chat container styling */
    .chat-container {
        background: var(--bg-primary);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 1.5rem;
        margin-bottom: 1rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid var(--border-color);
    }
    
    /* Message styling */
    .message {
        margin-bottom: 1.5rem;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin-left: 20%;
        box-shadow: var(--shadow);
        position: relative;
    }
    
    .user-message::before {
        content: "👤";
        position: absolute;
        right: -2.5rem;
        top: 0.5rem;
        font-size: 1.5rem;
        background: white;
        border-radius: 50%;
        padding: 0.3rem;
        box-shadow: var(--shadow);
    }
    
    .assistant-message {
        background: var(--bg-secondary);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin-right: 20%;
        border: 1px solid var(--border-color);
        position: relative;
    }
    
    .assistant-message::before {
        content: "🤖";
        position: absolute;
        left: -2.5rem;
        top: 0.5rem;
        font-size: 1.5rem;
        background: white;
        border-radius: 50%;
        padding: 0.3rem;
        box-shadow: var(--shadow);
    }
    
    .message-meta {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: var(--bg-secondary);
        border-radius: var(--border-radius);
        padding: 1rem;
    }
    
    /* Chat list styling */
    .chat-list {
        max-height: 300px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    
    .chat-item {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .chat-item:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow);
        border-color: var(--primary-color);
    }
    
    .chat-item.active {
        background: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    .chat-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .chat-preview {
        font-size: 0.75rem;
        opacity: 0.7;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .chat-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.7rem;
        margin-top: 0.3rem;
        opacity: 0.6;
    }
    
    .delete-chat {
        position: absolute;
        top: 0.3rem;
        right: 0.3rem;
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.3s ease;
        font-size: 0.7rem;
        color: var(--error-color);
    }
    
    .chat-item:hover .delete-chat {
        opacity: 1;
    }
    
    .delete-chat:hover {
        background: rgba(255, 107, 107, 0.2);
    }
    
    /* New chat button */
    .new-chat-btn {
        background: linear-gradient(135deg, var(--success-color), var(--primary-color));
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.8rem 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .new-chat-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(67, 233, 123, 0.3);
    }
    
    /* Tab styling */
    .sidebar-tabs {
        display: flex;
        margin-bottom: 1rem;
        background: var(--bg-secondary);
        border-radius: var(--border-radius);
        padding: 0.2rem;
    }
    
    .sidebar-tab {
        flex: 1;
        padding: 0.6rem 1rem;
        text-align: center;
        cursor: pointer;
        border-radius: calc(var(--border-radius) - 0.2rem);
        transition: all 0.3s ease;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .sidebar-tab.active {
        background: var(--primary-color);
        color: white;
        box-shadow: var(--shadow);
    }
    
    .sidebar-tab:hover:not(.active) {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Quick examples styling */
    .example-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin-bottom: 0.8rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .example-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow);
        border-color: var(--primary-color);
    }
    
    .example-lang {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.3rem;
        letter-spacing: 0.5px;
    }
    
    .example-text {
        color: var(--text-primary);
        font-weight: 500;
        line-height: 1.4;
    }
    
    /* Status indicators */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(67, 233, 123, 0.1);
        color: var(--success-color);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(67, 233, 123, 0.2);
        margin-bottom: 1rem;
    }
    
    .status-dot {
        width: 6px;
        height: 6px;
        background: var(--success-color);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Input styling */
    .stTextArea textarea {
        border-radius: var(--border-radius);
        border: 2px solid var(--border-color);
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: var(--border-radius);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.7rem 2rem;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox select {
        border-radius: var(--border-radius);
        border: 2px solid var(--border-color);
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Welcome screen */
    /* Chat header */
    .chat-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    .chat-header h1 {
        margin: 0 0 10px 0;
        font-size: 2em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .chat-header p {
        margin: 0;
        color: #888;
        font-size: 0.9em;
    }
    
    .welcome-container {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: var(--border-radius);
        margin: 2rem 0;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .assistant-message {
            margin-left: 0;
            margin-right: 0;
        }
        
        .user-message::before, .assistant-message::before {
            display: none;
        }
        
        .app-title {
            font-size: 2rem;
        }
        
        .language-badges {
            gap: 0.3rem;
        }
        
        .language-badge {
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
        }
    }
    
    /* Hide Streamlit branding but keep navigation menu */
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'active_chat_id' not in st.session_state:
        st.session_state.active_chat_id = None
    if 'api_status' not in st.session_state:
        st.session_state.api_status = 'checking'
    if 'chat_counter' not in st.session_state:
        st.session_state.chat_counter = 0
    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'chats'  # 'chats', 'settings', or 'learning'

# Check API status
@st.cache_data(ttl=10)  # Reduced cache time for faster updates
def check_api_status():
    # Try multiple ports where the API might be running (expanded range)
    ports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 3000, 5000]
    for port in ports:
        try:
            response = requests.get(f'http://localhost:{port}/health', timeout=3)
            if response.status_code == 200:
                return True, port
        except Exception as e:
            # Silently continue to next port
            continue
    return False, None

def get_api_url():
    """Get the correct API URL"""
    is_online, port = check_api_status()
    if is_online and port:
        return f'http://localhost:{port}'
    return 'http://localhost:8001'  # Default fallback

# Chat management functions
def create_new_chat():
    """Create a new chat session"""
    st.session_state.chat_counter += 1
    chat_id = f"chat_{st.session_state.chat_counter}"
    
    st.session_state.chats[chat_id] = {
        'id': chat_id,
        'title': f'New Chat {st.session_state.chat_counter}',
        'messages': [],
        'conversation_id': None,
        'created_at': datetime.datetime.now(),
        'updated_at': datetime.datetime.now(),
        'total_messages': 0,
        'total_tokens': 0,
        'language': 'auto'
    }
    
    st.session_state.active_chat_id = chat_id
    return chat_id

def get_active_chat():
    """Get the currently active chat"""
    if not st.session_state.active_chat_id or st.session_state.active_chat_id not in st.session_state.chats:
        # Create first chat if none exists
        if not st.session_state.chats:
            create_new_chat()
        else:
            # Select first available chat
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
    
    return st.session_state.chats[st.session_state.active_chat_id]

def update_chat_title(chat_id, new_title):
    """Update chat title"""
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]['title'] = new_title
        st.session_state.chats[chat_id]['updated_at'] = datetime.datetime.now()

def delete_chat(chat_id):
    """Delete a chat"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        
        # Switch to another chat or create new one
        if st.session_state.chats:
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
        else:
            create_new_chat()

def add_message_to_chat(chat_id, role, content, **kwargs):
    """Add a message to a specific chat"""
    if chat_id in st.session_state.chats:
        chat = st.session_state.chats[chat_id]
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.datetime.now().strftime("%H:%M"),
            **kwargs
        }
        chat['messages'].append(message)
        chat['updated_at'] = datetime.datetime.now()
        chat['total_messages'] += 1
        
        # Auto-update title based on first user message
        if role == 'user' and len(chat['messages']) == 1:
            title = content[:30] + "..." if len(content) > 30 else content
            chat['title'] = title

def get_chat_summary(chat):
    """Get a summary of the chat for display"""
    if not chat['messages']:
        return "No messages yet"
    
    last_message = chat['messages'][-1]
    if last_message['role'] == 'user':
        preview = f"You: {last_message['content'][:40]}"
    else:
        preview = f"AI: {last_message['content'][:40]}"
    
    return preview + "..." if len(last_message['content']) > 40 else preview

# Language detection and examples
LANGUAGE_CONFIG = {
    'auto': {
        'name': 'Auto-detect',
        'flag': '🌐',
        'examples': [
            'Hello, how are you today?',
            'Habari yako? Hujambo?',
            'Wĩ atĩa? Nĩ ũguo mwega?',
            'Inadi? Imiyau nade?'
        ]
    },
    'en': {
        'name': 'English',
        'flag': '🇺🇸',
        'examples': [
            'Hello, how can you help me?',
            'Tell me about your capabilities',
            'What languages do you speak?',
            'Can you help me learn?'
        ]
    },
    'sw': {
        'name': 'Kiswahili',
        'flag': '🇰🇪',
        'examples': [
            'Habari yako? Unaweza kunisaidia?',
            'Niambie kuhusu uwezo wako',
            'Nakupenda sana wewe',
            'Unaongea lugha gani?'
        ]
    },
    'ki': {
        'name': 'Kikuyu',
        'flag': '🇰🇪',
        'examples': [
            'Wĩ atĩa? Ũngĩndeithagia?',
            'Njĩra cia gũthoma ciũgano',
            'Nĩngũkwenda mũno',
            'Wĩ mũtaare wa ciũgano?'
        ]
    },
    'luo': {
        'name': 'Luo',
        'flag': '🇰🇪',
        'examples': [
            'Inadi? Inyalo konya nadi?',
            'Nyisa kuom tekoni magi',
            'Aheri miwuoro matek',
            'Iwacho dhok mage?'
        ]
    }
}

# Main app function
def export_all_chats():
    """Export all conversations to JSON."""
    try:
        export_data = {
            'exported_at': datetime.datetime.now().isoformat(),
            'chats': st.session_state.chats,
            'total_chats': len(st.session_state.chats),
            'statistics': {
                'total_messages': sum(chat['total_messages'] for chat in st.session_state.chats.values()),
                'total_tokens': sum(chat['total_tokens'] for chat in st.session_state.chats.values()),
            }
        }
        
        # Convert to JSON
        json_str = json.dumps(export_data, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="📥 Download All Chats",
            data=json_str,
            file_name=f"all_chats_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("All chat exports prepared for download!")
        
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def main():
    # Load custom CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Check API status
    api_online, api_port = check_api_status()
    st.session_state.api_status = 'online' if api_online else 'offline'
    st.session_state.api_url = get_api_url()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🧠 Trilingual AI Assistant</div>
        <div class="app-subtitle">
            Intelligent conversations in English, Kiswahili, Kikuyu, and Luo
        </div>
        <div class="language-badges">
            <span class="language-badge">🇺🇸 English</span>
            <span class="language-badge">🇰🇪 Kiswahili</span>
            <span class="language-badge">🇰🇪 Kikuyu</span>
            <span class="language-badge">🇰🇪 Luo</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with simplified navigation
    with st.sidebar:
        # API Status
        if api_online:
            st.success(f"✅ API Online (Port {api_port})")
        else:
            st.error("❌ API Offline")
        
        st.markdown("---")
        
        # Chat selection dropdown
        st.subheader("💬 Chat Selection")
        
        if st.session_state.chats:
            # Create options for dropdown
            chat_options = {}
            for chat_id, chat in st.session_state.chats.items():
                title = chat['title'][:30] + "..." if len(chat['title']) > 30 else chat['title']
                chat_options[title] = chat_id
            
            # Get current selection
            current_chat = get_active_chat()
            current_title = current_chat['title'][:30] + "..." if len(current_chat['title']) > 30 else current_chat['title']
            
            selected_chat_title = st.selectbox(
                "Select Chat:",
                options=list(chat_options.keys()),
                index=list(chat_options.keys()).index(current_title) if current_title in chat_options else 0,
                key="chat_selector"
            )
            
            # Update active chat if selection changed
            selected_chat_id = chat_options[selected_chat_title]
            if selected_chat_id != st.session_state.active_chat_id:
                st.session_state.active_chat_id = selected_chat_id
                st.rerun()
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()
        
        # Delete current chat button (only if more than one chat exists)
        if st.session_state.chats and len(st.session_state.chats) > 1:
            if st.button("🗑️ Delete Current Chat", use_container_width=True):
                delete_chat(st.session_state.active_chat_id)
                st.rerun()
        
        st.markdown("---")
        
        # Language selection
        st.subheader("🌐 Language Settings")
        
        # Get current chat for language preference
        active_chat = get_active_chat()
        current_language = active_chat.get('language', 'auto')
        
        selected_language = st.selectbox(
            "Response Language:",
            options=list(LANGUAGE_CONFIG.keys()),
            format_func=lambda x: f"{LANGUAGE_CONFIG[x]['flag']} {LANGUAGE_CONFIG[x]['name']}",
            index=list(LANGUAGE_CONFIG.keys()).index(current_language) if current_language in LANGUAGE_CONFIG else 0,
            key="language_selector"
        )
        
        # Update chat language preference
        if selected_language != current_language:
            active_chat['language'] = selected_language
        
        st.markdown("---")
        
        # Settings in expandable section
        with st.expander("⚙️ Settings"):
            # Temperature control
            temperature = st.slider("🌡️ Creativity", 0.1, 1.0, 0.7, 0.1, key="temperature_setting")
            
            # Quick examples
            st.markdown("**💡 Quick Examples:**")
            examples = LANGUAGE_CONFIG[selected_language]['examples']
            for i, example in enumerate(examples[:2]):  # Limit to 2 examples
                if st.button(f"📝 {example[:25]}...", key=f"example_{i}", use_container_width=True):
                    st.session_state.example_clicked = example
            
            # Export chat
            if st.button("💾 Export Current Chat", use_container_width=True):
                export_chat()
            
            # Clear chat
            if st.button("🗑️ Clear Messages", use_container_width=True):
                active_chat = get_active_chat()
                active_chat['messages'] = []
                active_chat['total_messages'] = 0
                active_chat['total_tokens'] = 0
                st.rerun()
        
        # Chat statistics in expandable section
        if st.session_state.chats:
            with st.expander("📊 Statistics"):
                total_messages = sum(chat['total_messages'] for chat in st.session_state.chats.values())
                total_tokens = sum(chat['total_tokens'] for chat in st.session_state.chats.values())
                
                st.metric("Total Chats", len(st.session_state.chats))
                st.metric("Total Messages", total_messages)
                st.metric("Total Tokens", total_tokens)
    
    # Main content area
    active_chat = get_active_chat()
    
    # Simple chat title
    st.subheader(f"💬 {active_chat['title']}")
    
    # Display messages or welcome screen
    messages = active_chat.get('messages', [])
    
    if not messages:
        # Simple welcome message
        st.info("� Welcome! Start a conversation in any language. The AI will respond in your selected language.")
    else:
        # Display chat messages
        for message in messages:
            if message["role"] == "user":
                with st.container():
                    st.markdown(f"""
                    <div class="message">
                        <div class="user-message">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Simple metadata
                    st.caption(f"👤 You • {message.get('timestamp', '')}")
            else:
                with st.container():
                    st.markdown(f"""
                    <div class="message">
                        <div class="assistant-message">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Simple metadata
                    lang = message.get('language', 'Unknown')
                    confidence = message.get('confidence', 0)
                    st.caption(f"🤖 AI • {lang} • {confidence:.0%} confidence • {message.get('timestamp', '')}")
    
    # Simple message input
    st.markdown("---")
    
    # Check for example click
    if hasattr(st.session_state, 'example_clicked'):
        user_input = st.session_state.example_clicked
        del st.session_state.example_clicked
    else:
        user_input = None
    
    # Message input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        message = st.text_area(
            "💬 Your message:",
            value=user_input or "",
            placeholder=f"Type in {LANGUAGE_CONFIG[selected_language]['name']}...",
            height=80,
            key="user_message"
        )
    
    with col2:
        st.write("")  # Add some spacing
        st.write("")  # Add some spacing
        send_clicked = st.button("Send", type="primary")
    
    # Handle message sending
    if send_clicked and message.strip():
        if not api_online:
            st.error("❌ API server is offline. Please start the server first.")
            return
        
        # Add user message
        add_message_to_chat(
            st.session_state.active_chat_id, 
            "user", 
            message,
            timestamp=datetime.datetime.now().strftime("%H:%M")
        )
        
        # Clear the text box by deleting the session state key
        if "user_message" in st.session_state:
            del st.session_state["user_message"]
        
        # Send to API
        with st.spinner("🤔 AI is thinking..."):
            try:
                # Convert language selection to API format
                language_mapping = {
                    'auto': 'auto',
                    'en': 'english',
                    'sw': 'kiswahili',
                    'ki': 'kikuyu',
                    'luo': 'luo'
                }
                
                api_language = language_mapping.get(selected_language, 'auto')
                
                response = requests.post(f'{st.session_state.api_url}/chat', 
                    json={
                        "message": message,
                        "language": api_language,
                        "conversation_id": active_chat.get('conversation_id'),
                        "max_length": 100,
                        "temperature": temperature
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update conversation ID and tokens
                    if data.get('conversation_id'):
                        active_chat['conversation_id'] = data['conversation_id']
                        active_chat['total_tokens'] += data.get('tokens_generated', 0)
                    
                    # Add assistant response
                    add_message_to_chat(
                        st.session_state.active_chat_id,
                        "assistant",
                        data.get('response', 'No response'),
                        timestamp=datetime.datetime.now().strftime("%H:%M"),
                        language=data.get('language_detected', 'Unknown'),
                        confidence=data.get('confidence', 0),
                        tokens=data.get('tokens_generated', 0)
                    )
                    
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    
            except requests.RequestException as e:
                st.error(f"❌ Connection Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def export_chat():
    """Export current active chat history"""
    active_chat = get_active_chat()
    messages = active_chat.get('messages', [])
    
    if not messages:
        st.warning("No messages to export in current chat")
        return
    
    export_data = {
        "chat_id": st.session_state.active_chat_id,
        "chat_title": active_chat['title'],
        "timestamp": datetime.datetime.now().isoformat(),
        "created_at": active_chat['created_at'].isoformat(),
        "updated_at": active_chat['updated_at'].isoformat(),
        "total_messages": active_chat['total_messages'],
        "total_tokens": active_chat['total_tokens'],
        "conversation_id": active_chat.get('conversation_id'),
        "messages": messages
    }
    
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    st.download_button(
        label="💾 Download Current Chat",
        data=json_str,
        file_name=f"chat-{active_chat['title'][:20]}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
