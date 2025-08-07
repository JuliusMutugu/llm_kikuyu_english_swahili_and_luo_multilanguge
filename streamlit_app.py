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
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
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
        st.session_state.sidebar_state = 'chats'  # 'chats' or 'settings'

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
    
    # Sidebar with tabs
    with st.sidebar:
        # API Status
        st.markdown(f"""
        <div class="status-online">
            <div class="status-dot"></div>
            AI Assistant {'Online' if api_online else 'Offline'} {f'(Port {api_port})' if api_online and api_port else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar tabs
        col_tab1, col_tab2 = st.columns(2)
        with col_tab1:
            if st.button("💬 Chats", use_container_width=True, 
                        type="primary" if st.session_state.sidebar_state == 'chats' else "secondary"):
                st.session_state.sidebar_state = 'chats'
        
        with col_tab2:
            if st.button("⚙️ Settings", use_container_width=True,
                        type="primary" if st.session_state.sidebar_state == 'settings' else "secondary"):
                st.session_state.sidebar_state = 'settings'
        
        st.markdown("---")
        
        if st.session_state.sidebar_state == 'chats':
            # Chat management section
            st.subheader("💬 Conversations")
            
            # New chat button
            if st.button("➕ New Chat", use_container_width=True, type="primary"):
                create_new_chat()
                st.rerun()
            
            # Chat list
            if st.session_state.chats:
                st.markdown("**Your Chats:**")
                
                for chat_id, chat in sorted(st.session_state.chats.items(), 
                                          key=lambda x: x[1]['updated_at'], reverse=True):
                    col_chat, col_delete = st.columns([4, 1])
                    
                    with col_chat:
                        # Chat selection
                        is_active = chat_id == st.session_state.active_chat_id
                        chat_summary = get_chat_summary(chat)
                        
                        if st.button(
                            f"{'🔵 ' if is_active else '⚪ '}{chat['title'][:25]}..." if len(chat['title']) > 25 else f"{'🔵 ' if is_active else '⚪ '}{chat['title']}",
                            key=f"chat_select_{chat_id}",
                            use_container_width=True,
                            help=f"{chat_summary}\n{chat['total_messages']} messages • {chat['total_tokens']} tokens"
                        ):
                            st.session_state.active_chat_id = chat_id
                            st.rerun()
                    
                    with col_delete:
                        if len(st.session_state.chats) > 1:  # Don't allow deleting the last chat
                            if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                                delete_chat(chat_id)
                                st.rerun()
            
            # Chat statistics
            if st.session_state.chats:
                total_messages = sum(chat['total_messages'] for chat in st.session_state.chats.values())
                total_tokens = sum(chat['total_tokens'] for chat in st.session_state.chats.values())
                
                st.markdown("---")
                st.markdown("**Overall Statistics**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Chats", len(st.session_state.chats))
                with col2:
                    st.metric("Total Messages", total_messages)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Total Tokens", total_tokens)
                with col4:
                    st.metric("Avg Msgs/Chat", f"{total_messages/len(st.session_state.chats):.1f}")
        
        else:  # Settings tab
            # Language selection
            st.subheader("🌍 Language Settings")
            selected_language = st.selectbox(
                "Choose language preference:",
                options=list(LANGUAGE_CONFIG.keys()),
                format_func=lambda x: f"{LANGUAGE_CONFIG[x]['flag']} {LANGUAGE_CONFIG[x]['name']}",
                index=0
            )
            
            # Model settings
            st.subheader("⚙️ Model Settings")
            temperature = st.slider("Temperature (creativity)", 0.1, 1.0, 0.7, 0.1)
            max_length = st.slider("Response length", 50, 200, 100, 10)
            
            # Quick examples
            st.subheader("💡 Quick Examples")
            examples = LANGUAGE_CONFIG[selected_language]['examples']
            
            for i, example in enumerate(examples):
                if st.button(f"📝 {example[:30]}...", key=f"example_{i}", use_container_width=True):
                    st.session_state.example_clicked = example
            
            # Actions
            st.subheader("🛠️ Actions")
            if st.button("🗑️ Clear Active Chat", use_container_width=True):
                if st.session_state.active_chat_id:
                    active_chat = get_active_chat()
                    active_chat['messages'] = []
                    active_chat['conversation_id'] = None
                    active_chat['total_messages'] = 0
                    active_chat['total_tokens'] = 0
                    active_chat['updated_at'] = datetime.datetime.now()
                    st.rerun()
            
            if st.button("💾 Export All Chats", use_container_width=True):
                export_all_chats()
            
            if st.button("🔄 Refresh API Status", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            # Debug section
            with st.expander("🔧 Debug Info", expanded=False):
                if st.button("🧪 Test API Connection", use_container_width=True):
                    st.write("Testing API connections...")
                    ports_to_test = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009]
                    
                    for port in ports_to_test:
                        try:
                            response = requests.get(f'http://localhost:{port}/health', timeout=2)
                            if response.status_code == 200:
                                st.success(f"✅ Port {port}: API responding")
                            else:
                                st.error(f"❌ Port {port}: Status {response.status_code}")
                        except requests.exceptions.ConnectionError:
                            st.warning(f"⚠️ Port {port}: Connection refused")
                        except Exception as e:
                            st.error(f"❌ Port {port}: {str(e)}")
                
                st.write(f"**Current API URL:** {st.session_state.api_url}")
                st.write(f"**API Status:** {st.session_state.api_status}")
    
    # Main content area
    active_chat = get_active_chat()
    
    # Chat header
    st.markdown(f"""
    <div class="chat-header">
        <h1>💬 {active_chat['title']}</h1>
        <p>{active_chat['total_messages']} messages • {active_chat['total_tokens']} tokens • Created {active_chat['created_at'].strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Edit chat title
    with st.expander("✏️ Edit Chat Title", expanded=False):
        new_title = st.text_input("Chat title:", value=active_chat['title'], key="title_edit")
        if st.button("Update Title") and new_title != active_chat['title']:
            update_chat_title(st.session_state.active_chat_id, new_title)
            st.success(f"Title updated to: {new_title}")
            st.rerun()
    
    # Display messages or welcome screen
    messages = active_chat.get('messages', [])
    
    if not messages:
        # Welcome screen for new chat
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">💬</div>
            <div class="welcome-title">Welcome to Multilingual AI</div>
            <div class="welcome-subtitle">
                Start a conversation in any language. I understand and respond in English, 
                Kiswahili, Kikuyu, and Luo. Try the examples on the left or type your own message!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat messages
        for message in messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message">
                    <div class="user-message">
                        {message["content"]}
                        <div class="message-meta">
                            <span>You</span>
                            <span>{message.get('timestamp', '')}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message">
                    <div class="assistant-message">
                        {message["content"]}
                        <div class="message-meta">
                            <span>🤖 AI Assistant • {message.get('language', 'Unknown')} • {message.get('confidence', 0):.0%} confidence</span>
                            <span>{message.get('timestamp', '')} • {message.get('tokens', 0)} tokens</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    
    # Check for example click
    if hasattr(st.session_state, 'example_clicked'):
        user_input = st.session_state.example_clicked
        del st.session_state.example_clicked
    else:
        user_input = None
    
    # Message input
    col_input, col_send = st.columns([4, 1])
    
    with col_input:
        message = st.text_area(
            "Type your message:",
            value=user_input or "",
            placeholder=f"Type in any language... (English, Kiswahili, Kikuyu, Luo)",
            height=100,
            key="user_message"
        )
    
    with col_send:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        send_clicked = st.button("🚀 Send", use_container_width=True, type="primary")
    
    # Handle message sending
    if send_clicked and message.strip():
        if not api_online:
            st.error("❌ API server is offline. Please start the server first.")
            return
        
        # Add user message to active chat
        timestamp = datetime.datetime.now().strftime("%H:%M")
        add_message_to_chat(
            st.session_state.active_chat_id, 
            "user", 
            message,
            timestamp=timestamp
        )
        
        # Send to API
        with st.spinner("🤔 Thinking..."):
            try:
                # Get language and model settings from sidebar
                selected_language = "english"  # Default
                temperature = 0.7  # Default
                max_length = 100  # Default
                
                response = requests.post(f'{st.session_state.api_url}/chat', 
                    json={
                        "message": message,
                        "language": selected_language,
                        "conversation_id": active_chat.get('conversation_id'),
                        "max_length": max_length,
                        "temperature": temperature
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update conversation ID if needed
                    if data.get('conversation_id'):
                        active_chat['conversation_id'] = data['conversation_id']
                    
                    # Add assistant response to active chat
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
                st.error(f"❌ Unexpected Error: {str(e)}")
    
    # API Status in main area
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        if api_online:
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            st.info("Start the API server:\n```bash\npython multi_model_api.py\n```")
    
    with col_status2:
        # Current chat info
        if st.session_state.chats:
            st.info(f"""
            **Current Chat**  
            📊 {len(st.session_state.chats)} total chats  
            💬 {active_chat['total_messages']} messages in current  
            🎯 {active_chat['total_tokens']} tokens in current
            """)

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
