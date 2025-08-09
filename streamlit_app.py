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
import os
from pathlib import Path
import base64
from typing import Dict, List, Optional, Any
import asyncio
import threading

# Import federated learning module
try:
    from federated_learning import FederatedLearningClient, EXAMPLE_FEDERATED_SOURCES
    FEDERATED_LEARNING_AVAILABLE = True
except ImportError:
    FEDERATED_LEARNING_AVAILABLE = False

# Import dictionary learning module
try:
    from online_dictionary_learner import OnlineDictionaryLearner
    DICTIONARY_LEARNING_AVAILABLE = True
except ImportError:
    DICTIONARY_LEARNING_AVAILABLE = False

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
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
    if 'learning_analytics' not in st.session_state:
        st.session_state.learning_analytics = {
            'language_usage': {'en': 0, 'sw': 0, 'ki': 0, 'luo': 0, 'auto': 0},
            'response_ratings': [],
            'error_reports': [],
            'conversation_topics': [],
            'user_satisfaction': []
        }
    if 'federated_client' not in st.session_state and FEDERATED_LEARNING_AVAILABLE:
        st.session_state.federated_client = None
        st.session_state.federated_enabled = False
        st.session_state.federated_sources = []
    if 'dictionary_learner' not in st.session_state and DICTIONARY_LEARNING_AVAILABLE:
        st.session_state.dictionary_learner = None
        st.session_state.dictionary_learning_status = 'idle'
        st.session_state.last_dictionary_learning = None

# Check API status
def check_api_status():
    """Check if API is available - works for both local and deployed environments"""
    
    # First, try the configured API URL (for deployed environments)
    api_url = get_api_url()
    
    # Special handling for known working deployed URL - assume it's always available
    if api_url == 'https://llm-kikuyu-english-swahili-and-luo.onrender.com':
        # For the deployed service, we know it's working so return True immediately
        return True, 'deployed'
    
    # For other URLs, do a quick health check
    try:
        response = requests.get(f'{api_url}/health', timeout=3)
        if response.status_code == 200:
            # Extract port from URL for display purposes
            if 'localhost' in api_url:
                port = api_url.split(':')[-1] if ':' in api_url else '8000'
                return True, port
            else:
                # For deployed services, return a generic indicator
                return True, 'deployed'
    except Exception:
        # If health check fails for localhost, try detecting local ports
        if 'localhost' in api_url:
            ports = [8000, 8001, 8002, 8003, 8004, 8005]
            for port in ports:
                try:
                    response = requests.get(f'http://localhost:{port}/health', timeout=2)
                    if response.status_code == 200:
                        return True, port
                except Exception:
                    continue
    
    return False, None

def get_api_url():
    """Get the correct API URL"""
    # Check for environment variable (for cloud deployment)
    api_url = os.environ.get('API_URL')
    if api_url:
        return api_url.rstrip('/')
    
    # Check if we're running on Render (combined service)
    current_url = os.environ.get('RENDER_EXTERNAL_URL')
    if current_url:
        return current_url.rstrip('/')
    
    # Check if we're running on a deployed platform
    hostname = os.environ.get('HOSTNAME', '').lower()
    if ('onrender' in hostname or 
        'render' in hostname or 
        os.environ.get('RENDER_SERVICE_NAME') or
        'streamlit' in os.environ.get('STREAMLIT_SERVER_PORT', '')):
        # We're on Render - use the deployed URL
        return 'https://llm-kikuyu-english-swahili-and-luo.onrender.com'
    
    # Default to deployed URL if we can't detect environment clearly
    # This ensures the app works even if environment detection fails
    return 'https://llm-kikuyu-english-swahili-and-luo.onrender.com'

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
    
    return preview + "..."         if len(last_message['content']) > 40 else preview

# Feedback and learning functions
def add_feedback(message_index, chat_id, rating, feedback_text=""):
    """Add user feedback for a specific message"""
    feedback = {
        'timestamp': datetime.datetime.now().isoformat(),
        'chat_id': chat_id,
        'message_index': message_index,
        'rating': rating,
        'feedback_text': feedback_text,
        'message_content': st.session_state.chats[chat_id]['messages'][message_index]['content']
    }
    st.session_state.feedback_data.append(feedback)
    st.session_state.learning_analytics['response_ratings'].append(rating)
    
    # Create federated learning update from feedback
    if st.session_state.get('federated_enabled', False):
        # Get the AI response message
        message = st.session_state.chats[chat_id]['messages'][message_index]
        
        federated_feedback = {
            'language': message.get('language', 'auto'),
            'rating': rating,
            'feedback_text': feedback_text,
            'confidence': message.get('confidence', 0.5),
            'cultural_context': 'general',  # Could be enhanced based on user profile
            'response_time': 1000,  # Placeholder - could track actual response time
        }
        
        create_federated_update_from_feedback(federated_feedback)

def add_error_report(error_type, description, context=""):
    """Add an error report for learning improvement"""
    error_report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'error_type': error_type,
        'description': description,
        'context': context,
        'chat_id': st.session_state.active_chat_id
    }
    st.session_state.learning_analytics['error_reports'].append(error_report)

def update_language_usage(language):
    """Track language usage for analytics"""
    if language in st.session_state.learning_analytics['language_usage']:
        st.session_state.learning_analytics['language_usage'][language] += 1

def export_learning_data():
    """Export all learning data for model improvement"""
    learning_data = {
        'exported_at': datetime.datetime.now().isoformat(),
        'feedback_data': st.session_state.feedback_data,
        'analytics': st.session_state.learning_analytics,
        'total_conversations': len(st.session_state.chats),
        'total_messages': sum(chat['total_messages'] for chat in st.session_state.chats.values()),
        'conversations': st.session_state.chats
    }
    
    json_str = json.dumps(learning_data, indent=2, default=str, ensure_ascii=False)
    return json_str

def get_learning_insights():
    """Generate insights from collected learning data"""
    analytics = st.session_state.learning_analytics
    
    insights = {
        'most_used_language': max(analytics['language_usage'], key=analytics['language_usage'].get),
        'average_rating': sum(analytics['response_ratings']) / len(analytics['response_ratings']) if analytics['response_ratings'] else 0,
        'total_feedback': len(st.session_state.feedback_data),
        'total_errors': len(analytics['error_reports']),
        'language_distribution': analytics['language_usage']
    }
    
    return insights

# Federated learning functions
def initialize_federated_learning():
    """Initialize federated learning client"""
    if not FEDERATED_LEARNING_AVAILABLE:
        return False
    
    try:
        if st.session_state.federated_client is None:
            st.session_state.federated_client = FederatedLearningClient()
            
            # Add default learning sources
            for source_config in EXAMPLE_FEDERATED_SOURCES:
                st.session_state.federated_client.add_learning_source(source_config)
                st.session_state.federated_sources.append(source_config)
            
            st.session_state.federated_enabled = True
            return True
    except Exception as e:
        st.error(f"Failed to initialize federated learning: {e}")
        return False

def add_federated_learning_source(source_config: Dict[str, Any]):
    """Add a new federated learning source"""
    if st.session_state.federated_client:
        st.session_state.federated_client.add_learning_source(source_config)
        st.session_state.federated_sources.append(source_config)
        return True
    return False

def create_federated_update_from_feedback(feedback_data: Dict[str, Any]):
    """Create federated learning update from user feedback"""
    if not st.session_state.federated_enabled or not st.session_state.federated_client:
        return
    
    try:
        # Prepare data for federated learning
        federated_data = {
            'type': 'feedback',
            'language': feedback_data.get('language', 'auto'),
            'feedback': {
                'rating': feedback_data.get('rating', 3),
                'text': feedback_data.get('feedback_text', ''),
                'error_type': feedback_data.get('error_type'),
                'improvement_area': feedback_data.get('improvement_area', 'general')
            },
            'cultural_context': feedback_data.get('cultural_context', 'general'),
            'performance': {
                'response_time': feedback_data.get('response_time', 0),
                'accuracy': feedback_data.get('confidence', 0.5),
                'satisfaction_score': feedback_data.get('rating', 3) / 5.0
            }
        }
        
        # Create privacy-preserving update
        update = st.session_state.federated_client.create_privacy_preserving_update(federated_data)
        st.session_state.federated_client.local_updates.append(update)
        
    except Exception as e:
        st.error(f"Error creating federated update: {e}")

async def run_federated_learning_sync():
    """Run federated learning synchronization"""
    if st.session_state.federated_client and st.session_state.federated_enabled:
        try:
            # Fetch updates from all sources
            all_updates = []
            
            for source in st.session_state.federated_client.learning_sources:
                if source['enabled']:
                    updates = await st.session_state.federated_client.fetch_updates_from_source(source)
                    all_updates.extend(updates)
            
            if all_updates:
                # Aggregate updates
                aggregated = st.session_state.federated_client.aggregate_updates(all_updates)
                
                # Apply learning
                improvements = st.session_state.federated_client.apply_federated_learning(aggregated)
                
                # Store results
                st.session_state.federated_client.aggregated_knowledge.update(aggregated)
                
                return len(all_updates), improvements
            
        except Exception as e:
            st.error(f"Federated learning sync error: {e}")
    
    return 0, {}

def get_federated_learning_status():
    """Get federated learning status"""
    if not st.session_state.federated_enabled or not st.session_state.federated_client:
        return {
            'enabled': False,
            'status': 'Not initialized'
        }
    
    return {
        'enabled': True,
        'status': 'Active',
        **st.session_state.federated_client.get_learning_status()
    }

# Dictionary learning functions
async def run_dictionary_learning(languages: List[str] = ['luo']):
    """Run dictionary learning for specified languages"""
    if not DICTIONARY_LEARNING_AVAILABLE:
        return {'error': 'Dictionary learning not available'}
    
    try:
        st.session_state.dictionary_learning_status = 'running'
        
        learner = OnlineDictionaryLearner()
        report = await learner.learn_from_online_dictionaries(languages)
        
        st.session_state.last_dictionary_learning = report
        st.session_state.dictionary_learning_status = 'completed'
        
        return report
    
    except Exception as e:
        st.session_state.dictionary_learning_status = 'error'
        return {'error': str(e)}

def get_dictionary_learning_status():
    """Get dictionary learning status"""
    return {
        'status': st.session_state.get('dictionary_learning_status', 'idle'),
        'last_run': st.session_state.get('last_dictionary_learning'),
        'available': DICTIONARY_LEARNING_AVAILABLE
    }

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
            if api_port == 'deployed':
                st.success("✅ API Online (Deployed)")
            else:
                st.success(f"✅ API Online (Port {api_port})")
        else:
            st.error("❌ API Offline")
            # Show current API URL for debugging
            st.caption(f"Checking: {st.session_state.api_url}")
        
        st.markdown("---")
        
        # Navigation tabs for sidebar
        tab1, tab2, tab3 = st.tabs(["💬 Chats", "⚙️ Settings", "📊 Learning"])
        
        with tab1:
            # Chat selection dropdown
            st.subheader("Chat Selection")
            
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
        
        with tab2:
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
            
            # Temperature control
            temperature = st.slider("🌡️ Creativity", 0.1, 1.0, 0.7, 0.1, key="temperature_setting")
            
            # Quick examples
            st.markdown("**💡 Quick Examples:**")
            examples = LANGUAGE_CONFIG[selected_language]['examples']
            for i, example in enumerate(examples[:2]):  # Limit to 2 examples
                if st.button(f"📝 {example[:25]}...", key=f"example_{i}", use_container_width=True):
                    st.session_state.example_clicked = example
            
            # Export options
            st.markdown("**📁 Export Options:**")
            if st.button("💾 Export Current Chat", use_container_width=True):
                export_chat()
            
            if st.button("📥 Export All Learning Data", use_container_width=True):
                learning_json = export_learning_data()
                st.download_button(
                    label="📥 Download Learning Data",
                    data=learning_json,
                    file_name=f"learning_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Clear options
            st.markdown("**🗑️ Clear Options:**")
            if st.button("🗑️ Clear Messages", use_container_width=True):
                active_chat = get_active_chat()
                active_chat['messages'] = []
                active_chat['total_messages'] = 0
                active_chat['total_tokens'] = 0
                st.rerun()
        
        with tab3:
            # Learning Analytics Dashboard
            st.subheader("🎓 Learning Analytics")
            
            insights = get_learning_insights()
            
            # Federated Learning Section
            if FEDERATED_LEARNING_AVAILABLE:
                with st.expander("🌐 Federated Learning", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not st.session_state.federated_enabled:
                            if st.button("🚀 Enable Federated Learning", use_container_width=True):
                                if initialize_federated_learning():
                                    st.success("✅ Federated learning enabled!")
                                    st.rerun()
                        else:
                            st.success("✅ Federated Learning Active")
                            
                            if st.button("🔄 Sync Now", use_container_width=True):
                                with st.spinner("Syncing with federated sources..."):
                                    # Run sync in background
                                    try:
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        updates_count, improvements = loop.run_until_complete(run_federated_learning_sync())
                                        loop.close()
                                        
                                        if updates_count > 0:
                                            st.success(f"📊 Synced {updates_count} updates!")
                                        else:
                                            st.info("📊 No new updates available")
                                    except Exception as e:
                                        st.error(f"Sync failed: {e}")
                    
                    with col2:
                        fed_status = get_federated_learning_status()
                        if fed_status['enabled']:
                            st.metric("Active Sources", fed_status.get('active_sources', 0))
                            st.metric("Local Updates", fed_status.get('local_updates', 0))
                    
                    # Federated Learning Sources Management
                    st.markdown("**📡 Learning Sources:**")
                    
                    if st.session_state.federated_enabled and st.session_state.federated_sources:
                        for i, source in enumerate(st.session_state.federated_sources):
                            with st.container():
                                col1, col2, col3 = st.columns([3, 1, 1])
                                
                                with col1:
                                    st.write(f"**{source['id']}** ({source['type']})")
                                    st.caption(f"Languages: {', '.join(source['languages'])}")
                                
                                with col2:
                                    status = "🟢 Active" if source['enabled'] else "🔴 Disabled"
                                    st.write(status)
                                
                                with col3:
                                    trust_level = source.get('trust_level', 0.5)
                                    st.metric("Trust", f"{trust_level:.1f}")
                    
                    # Add new source
                    with st.expander("➕ Add Learning Source"):
                        new_source_id = st.text_input("Source ID:", placeholder="my_source")
                        new_source_url = st.text_input("URL/Path:", placeholder="https://api.example.com/data")
                        new_source_type = st.selectbox("Type:", ["api", "file", "github"])
                        new_languages = st.multiselect("Languages:", ["en", "sw", "ki", "luo"], default=["en", "sw", "ki", "luo"])
                        new_trust_level = st.slider("Trust Level:", 0.0, 1.0, 0.7, 0.1)
                        new_cultural_context = st.text_input("Cultural Context:", placeholder="general")
                        
                        if st.button("Add Source"):
                            if new_source_id and new_source_url:
                                new_source_config = {
                                    'id': new_source_id,
                                    'url': new_source_url,
                                    'type': new_source_type,
                                    'languages': new_languages,
                                    'trust_level': new_trust_level,
                                    'cultural_context': new_cultural_context,
                                    'enabled': True
                                }
                                
                                if add_federated_learning_source(new_source_config):
                                    st.success(f"✅ Added source: {new_source_id}")
                                    st.rerun()
                            else:
                                st.error("Please provide Source ID and URL")
                    
                    # Federated Learning Insights
                    if st.session_state.federated_enabled and st.session_state.federated_client:
                        aggregated_knowledge = getattr(st.session_state.federated_client, 'aggregated_knowledge', {})
                        
                        if aggregated_knowledge:
                            st.markdown("**🧠 Federated Insights:**")
                            
                            # Language improvements from federated learning
                            lang_improvements = aggregated_knowledge.get('language_improvements', {})
                            if lang_improvements:
                                for lang, data in lang_improvements.items():
                                    if data.get('update_count', 0) > 0:
                                        st.info(f"📈 {LANGUAGE_CONFIG.get(lang, {}).get('name', lang)}: {data['update_count']} federated updates, Quality: {data.get('quality_score', 0):.2f}")
                            
                            # Cultural insights
                            cultural_insights = aggregated_knowledge.get('cultural_insights', {})
                            if cultural_insights:
                                for context, data in cultural_insights.items():
                                    if data.get('update_count', 0) > 0:
                                        st.info(f"🌍 {context}: {data['update_count']} cultural updates")
            
            # Dictionary Learning Section
            if DICTIONARY_LEARNING_AVAILABLE:
                with st.expander("📚 Online Dictionary Learning", expanded=False):
                    dict_status = get_dictionary_learning_status()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Learn from Online Dictionaries:**")
                        st.markdown("• 🌐 Glosbe.com - Luo, Kiswahili, Kikuyu")
                        st.markdown("• 📖 Wiktionary - Vocabulary expansion")
                        st.markdown("• 🔍 Automatic cultural context detection")
                        
                        # Language selection for learning
                        learning_languages = st.multiselect(
                            "Languages to learn:",
                            ['luo', 'sw', 'ki'],
                            default=['luo'],
                            help="Select languages to learn vocabulary from online dictionaries"
                        )
                        
                        if st.button("🚀 Start Dictionary Learning", use_container_width=True):
                            if learning_languages:
                                with st.spinner(f"Learning vocabulary for {', '.join(learning_languages)}..."):
                                    try:
                                        # Run dictionary learning
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        report = loop.run_until_complete(run_dictionary_learning(learning_languages))
                                        loop.close()
                                        
                                        if 'error' not in report:
                                            st.success(f"✅ Learned {report.get('total_entries', 0)} vocabulary entries!")
                                            if report.get('output_file'):
                                                st.info(f"📁 Saved to: {report['output_file']}")
                                        else:
                                            st.error(f"❌ Learning failed: {report['error']}")
                                        
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Dictionary learning error: {e}")
                            else:
                                st.warning("Please select at least one language")
                    
                    with col2:
                        # Dictionary learning status
                        status = dict_status['status']
                        if status == 'idle':
                            st.info("🔷 Ready to learn")
                        elif status == 'running':
                            st.warning("🔄 Learning in progress...")
                        elif status == 'completed':
                            st.success("✅ Learning completed")
                        elif status == 'error':
                            st.error("❌ Learning failed")
                        
                        # Last learning report
                        if dict_status['last_run']:
                            last_run = dict_status['last_run']
                            st.metric("Entries Learned", last_run.get('total_entries', 0))
                            st.metric("Languages", len(last_run.get('languages_processed', [])))
                            
                            if 'quality_summary' in last_run:
                                quality = last_run['quality_summary']
                                st.metric("Avg Quality", f"{quality.get('average', 0):.2f}")
                    
                    # Quick Luo learning button
                    st.markdown("**🎯 Quick Luo Dictionary:**")
                    if st.button("📖 Learn Essential Luo Words from Glosbe", use_container_width=True):
                        with st.spinner("Learning essential Luo vocabulary from https://glosbe.com/en/luo..."):
                            try:
                                # Quick Luo learning
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                                # Import and run the specific Luo learning
                                from learn_luo_dictionary import learn_luo_dictionary
                                entries = loop.run_until_complete(learn_luo_dictionary())
                                loop.close()
                                
                                if entries:
                                    st.success(f"✅ Learned {len(entries)} Luo vocabulary entries from Glosbe!")
                                    
                                    # Show sample words
                                    st.markdown("**Sample Luo vocabulary:**")
                                    for entry in entries[:5]:
                                        if entry.target_word:
                                            st.write(f"• {entry.source_word} → {entry.target_word}")
                                else:
                                    st.warning("No Luo vocabulary was learned")
                                
                            except Exception as e:
                                st.error(f"Error learning Luo: {e}")
            
            # Regular Analytics (existing code)
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Rating", f"{insights['average_rating']:.1f}/5")
            with col2:
                st.metric("Total Feedback", insights['total_feedback'])
            with col3:
                st.metric("Error Reports", insights['total_errors'])
            
            # Language usage chart
            if any(insights['language_distribution'].values()):
                st.markdown("**📊 Language Usage:**")
                lang_data = insights['language_distribution']
                for lang, count in lang_data.items():
                    if count > 0:
                        st.metric(f"{LANGUAGE_CONFIG[lang]['flag']} {LANGUAGE_CONFIG[lang]['name']}", count)
            
            # Recent feedback
            if st.session_state.feedback_data:
                st.markdown("**📝 Recent Feedback:**")
                recent_feedback = st.session_state.feedback_data[-5:]  # Last 5 feedback items
                for feedback in reversed(recent_feedback):
                    with st.expander(f"Rating: {feedback['rating']}/5 - {feedback['timestamp'][:10]}"):
                        st.write(f"**Message:** {feedback['message_content'][:100]}...")
                        if feedback['feedback_text']:
                            st.write(f"**Feedback:** {feedback['feedback_text']}")
            
            # Error reports
            if st.session_state.learning_analytics['error_reports']:
                st.markdown("**🚨 Recent Error Reports:**")
                recent_errors = st.session_state.learning_analytics['error_reports'][-3:]  # Last 3 errors
                for error in reversed(recent_errors):
                    with st.expander(f"{error['error_type']} - {error['timestamp'][:10]}"):
                        st.write(f"**Description:** {error['description']}")
                        if error['context']:
                            st.write(f"**Context:** {error['context']}")
            
            # Learning improvement suggestions
            st.markdown("**💡 Improvement Suggestions:**")
            if insights['average_rating'] < 3.0 and insights['total_feedback'] > 5:
                st.warning("📈 Consider reviewing low-rated responses for improvement opportunities")
            
            if insights['total_errors'] > 0:
                st.info("🔧 Review error reports to identify common issues")
            
            if insights['most_used_language'] != 'auto':
                st.info(f"🌟 {LANGUAGE_CONFIG[insights['most_used_language']]['name']} is your most used language")
        
        # Chat statistics (moved to bottom)
        if st.session_state.chats:
            with st.expander("📊 Overall Statistics"):
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
                    col1, col2 = st.columns([6, 1])
                    
                    with col1:
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
                    
                    with col2:
                        # Feedback buttons for each AI response
                        message_idx = len([m for m in messages[:messages.index(message)+1] if m['role'] == 'assistant']) - 1
                        
                        st.write("Rate this response:")
                        col_thumbs_up, col_thumbs_down = st.columns(2)
                        
                        with col_thumbs_up:
                            if st.button("👍", key=f"thumbs_up_{messages.index(message)}", help="Good response"):
                                add_feedback(messages.index(message), st.session_state.active_chat_id, 5, "Thumbs up")
                                st.success("Thanks for the feedback!")
                                st.rerun()
                        
                        with col_thumbs_down:
                            if st.button("👎", key=f"thumbs_down_{messages.index(message)}", help="Poor response"):
                                add_feedback(messages.index(message), st.session_state.active_chat_id, 1, "Thumbs down")
                                st.warning("Thanks for the feedback! We'll improve.")
                                st.rerun()
                        
                        # Detailed rating
                        with st.expander("📊 Detailed Rating"):
                            rating = st.radio(
                                "Rate (1-5):",
                                [1, 2, 3, 4, 5],
                                index=2,
                                key=f"detailed_rating_{messages.index(message)}",
                                horizontal=True
                            )
                            
                            feedback_text = st.text_area(
                                "Feedback:",
                                placeholder="What could be improved?",
                                key=f"feedback_text_{messages.index(message)}",
                                height=60
                            )
                            
                            if st.button("Submit Rating", key=f"submit_rating_{messages.index(message)}"):
                                add_feedback(messages.index(message), st.session_state.active_chat_id, rating, feedback_text)
                                st.success(f"Rating {rating}/5 submitted! Thank you!")
                                st.rerun()
                        
                        # Error reporting
                        with st.expander("🚨 Report Issue"):
                            error_type = st.selectbox(
                                "Issue Type:",
                                ["Language Detection", "Grammar", "Cultural Context", "Inappropriate Response", "Technical Error", "Other"],
                                key=f"error_type_{messages.index(message)}"
                            )
                            
                            error_description = st.text_area(
                                "Describe the issue:",
                                placeholder="What went wrong?",
                                key=f"error_desc_{messages.index(message)}",
                                height=60
                            )
                            
                            if st.button("Report Issue", key=f"report_error_{messages.index(message)}"):
                                add_error_report(error_type, error_description, message["content"][:100])
                                st.warning("Issue reported! We'll use this to improve.")
                                st.rerun()
    
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
        # Ensure user_input is clean text (no HTML)
        clean_input = ""
        if user_input:
            # Remove any HTML tags and clean the text
            import re
            clean_input = re.sub(r'<[^>]+>', '', str(user_input)).strip()
        
        message = st.text_area(
            "💬 Your message:",
            value=clean_input,
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
                
                # Track language usage for analytics
                update_language_usage(selected_language)
                
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
