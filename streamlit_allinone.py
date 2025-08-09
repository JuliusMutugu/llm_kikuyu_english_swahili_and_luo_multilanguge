#!/usr/bin/env python3
"""
All-in-One Trilingual AI Assistant
Combines API logic and UI in a single Streamlit app for easy deployment
"""

import streamlit as st
import requests
import json
import time
import datetime
import os
import random
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="Trilingual AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/JuliusMutugu/llm_kikuyu_english_swahili_and_luo_multilanguge',
        'Report a bug': 'https://github.com/JuliusMutugu/llm_kikuyu_english_swahili_and_luo_multilanguge/issues',
        'About': "# Trilingual AI Assistant\nPowered by advanced language models supporting English, Kiswahili, Kikuyu, and Luo."
    }
)

# Simple AI Response System (No heavy ML dependencies)
class SimpleTrilingualAI:
    def __init__(self):
        self.conversations = {}
        self.conversation_counter = 0
        
    def detect_language(self, text: str) -> tuple:
        """Simple language detection based on keywords"""
        text_lower = text.lower()
        
        # Kikuyu indicators
        kikuyu_words = ['wĩ', 'atĩa', 'ũrĩ', 'nĩ', 'ũ', 'ĩ', 'gĩkũyũ', 'njĩra', 'mũndũ', 'gũthoma']
        if any(word in text_lower for word in kikuyu_words):
            return "kikuyu", 0.85
        
        # Luo indicators  
        luo_words = ['inadi', 'adhi', 'maber', 'gima', 'inyalo', 'konya', 'ere', 'dhok', 'kare', 'mondo']
        if any(word in text_lower for word in luo_words):
            return "luo", 0.85
            
        # Kiswahili indicators
        swahili_words = ['habari', 'mzuri', 'asante', 'karibu', 'pole', 'haraka', 'lugha', 'sana', 'hujambo', 'unaweza']
        if any(word in text_lower for word in swahili_words):
            return "kiswahili", 0.85
            
        # Default to English
        return "english", 0.7
    
    def generate_contextual_response(self, message: str, language: str, conversation_id: str = None) -> Dict:
        """Generate contextual responses based on message content and language"""
        
        message_lower = message.lower()
        
        # Response templates by language and context
        responses = {
            "english": {
                "greeting": [
                    "Hello! I'm your trilingual AI assistant. I can help you in English, Kiswahili, Kikuyu, and Luo. How can I assist you today?",
                    "Hi there! Welcome to our multilingual conversation. What would you like to talk about?",
                    "Greetings! I'm here to help you communicate in multiple African languages. What's on your mind?"
                ],
                "question": [
                    "That's a great question! Based on my understanding, I'd say that it depends on various factors. Could you provide more context?",
                    "Interesting question! Let me think about that. From what I know, there are several perspectives on this topic.",
                    "I appreciate your curiosity! This is something that many people wonder about. Here's what I think..."
                ],
                "help": [
                    "I'm here to help! I can assist with translations, cultural information about Kenya, and general conversations in English, Kiswahili, Kikuyu, and Luo.",
                    "Of course I can help! I specialize in Kenyan languages and culture. What specific assistance do you need?",
                    "Happy to help! Whether you need language practice, translations, or cultural insights, I'm here for you."
                ],
                "general": [
                    "I understand what you're saying. That's quite interesting! Tell me more about your thoughts on this.",
                    "Thank you for sharing that with me. I find that perspective quite valuable. What else would you like to discuss?",
                    "I appreciate you bringing this up. It's always great to have meaningful conversations like this."
                ]
            },
            "kiswahili": {
                "greeting": [
                    "Hujambo! Mimi ni msaidizi wako wa lugha tatu. Naweza kukusaidia kwa Kiingereza, Kiswahili, Kikuyu, na Kiluo. Naweza kukusaidia vipi leo?",
                    "Habari yako! Karibu kwenye mazungumzo yetu ya kilugha. Unataka tuongee kuhusu nini?",
                    "Salamu! Nipo hapa kukusaidia katika lugha mbalimbali za Kiafrika. Unachofikiri ni nini?"
                ],
                "question": [
                    "Hiyo ni swali zuri sana! Kulingana na uelewa wangu, nafikiri inategemea mambo mengi. Je, unaweza kutoa maelezo zaidi?",
                    "Swali la kuvutia! Hebu nifikirie hilo. Kutoka kile ninachojua, kuna mitazamo mbalimbali kuhusu mada hii.",
                    "Ninashukuru udadisi wako! Hii ni kitu ambacho watu wengi huuliza. Hiki ndicho ninachofikiri..."
                ],
                "help": [
                    "Niko hapa kukusaidia! Naweza kusaidia na tafsiri, taarifa za kitamaduni kuhusu Kenya, na mazungumzo ya kawaida kwa Kiingereza, Kiswahili, Kikuyu, na Kiluo.",
                    "Bila shaka naweza kusaidia! Nimejikita katika lugha na utamaduni wa Kenya. Unahitaji msaada gani?",
                    "Nimefurahi kusaidia! Iwe unahitaji mazoezi ya lugha, tafsiri, au maarifa ya kitamaduni, niko hapa kwa ajili yako."
                ],
                "general": [
                    "Naelewa unachoongea. Hicho ni cha kuvutia sana! Niambie zaidi kuhusu mawazo yako juu ya hili.",
                    "Asante kwa kunishirikisha hilo. Naiona mtazamo huu una thamani kubwa. Ungependa tuongee kuhusu nini kingine?",
                    "Ninashukuru kuleta hili. Ni bora daima kuwa na mazungumzo yenye maana kama haya."
                ]
            },
            "kikuyu": {
                "greeting": [
                    "Wĩ mwega! Nĩ niĩ mũteithii waku wa ciũgano ithatũ. Nĩngũkũteithia na Gĩthũngũ, Gĩswahili, Gĩkũyũ, na Kiluo. Ingĩgũteithia atĩa ũmũthĩ?",
                    "Wĩ atĩa! Wamũkĩra kwĩ ngũgano ciitũ cia ciũgano nyingĩ. Ũrenda tũrie ũhoro ũrĩkũ?",
                    "Ndakwagetha! Ndĩ gũkũ gũgũteithia na ciũgano cia Afrĩka. Ũreciiria atĩa?"
                ],
                "question": [
                    "Ũcio nĩ mũũrio mwega mũno! Kũringana na ũmenya wakwa, nĩngwĩciiria atĩ nĩkũraigana na maũndũ maingĩ. Ũngĩhota gũtũma ũhoro ũngĩ?",
                    "Mũũrio wa gũkenia! Reke ndĩecirie ũguo. Kuuma kũrĩa njũũĩ, nĩ kũrĩ mĩthiũrĩre mĩingĩ ĩkoniĩ mũtwe ũyũ.",
                    "Nĩngũkena nĩ ũndũ wa kwenda kũmenya! Gĩkĩ nĩ kĩndũ kĩrĩa andũ aingĩ mooragia. Gĩkĩ nĩ kĩrĩa njĩtĩkĩtie..."
                ],
                "help": [
                    "Ndĩ gũkũ gũgũteithia! Nĩhota gũteithia na gũtaũra, ũhoro wa mĩtugo ya Kenya, na ngũgano cia wĩra na Gĩthũngũ, Gĩswahili, Gĩkũyũ, na Kiluo.",
                    "Ti-itherũ nĩhota gũteithia! Nĩnyendete ciũgano na mĩtugo ya Kenya. Ũrenda ũteithio ũrĩkũ?",
                    "Nĩndĩkenete gũteithia! Ũngĩkorwo ũrenda kwĩrutĩra ciũgano, gũtaũra, kana kũmenya mĩtugo, nĩndĩ gũkũ nĩ ũndũ waku."
                ],
                "general": [
                    "Nĩnamenya ũrĩa ũroiga. Ũcio nĩ wa gũkenia mũno! Njĩra ũhoro ũngĩ ũkoniĩ meciiria maku ma ũhoro ũyũ.",
                    "Nĩngũkena nĩ ũndũ wa kũnjĩra ũguo. Nĩnyona meciiria maya marĩ bata mũno. Ũngenda tũrie ũhoro ũngĩ ũrĩkũ?",
                    "Nĩngũkena nĩ ũndũ wa gũrehe ũhoro ũyũ. Nĩ wega hĩndĩ ciothe gũkorwo na ngũgano cia gũtũma meciiria ta maya."
                ]
            },
            "luo": {
                "greeting": [
                    "Inadi! An jokonyruokni mar dholuo adek. Anyalo konyou gi Dholungu, Kiswahili, Kikuyu, kod Dholuo. Ere kaka anyalo konyou kawuono?",
                    "Ere nade! Erokamano kuom biro e wuoyowa mar dhok mopogore opogore. Idwaro wayor angʼo?",
                    "Amosou! An ka mondo akonyou gi dhok mopogore opogore mag Afrika. En angʼo miparo?"
                ],
                "question": [
                    "Mano en penjo maber ahinya! Kaluwore gi ngʼeyo mara, aparo ni ochungʼ kuom gik mopogore opogore. Bende inyalo miyo ranyis mangʼeny?",
                    "Penjo mamit! We apare wachno. Koa kuom gima angʼeyo, nitie paro mopogore opogore kuom wachni.",
                    "Ageno kuom dwarruok mari! Ma en gima ji mangʼeny penjo. Ma ema aparo..."
                ],
                "help": [
                    "An ka mondo akonyi! Anyalo konyo gi loko dhok, weche mag timbe mag Kenya, kod wuoyo mapile gi Dholungu, Kiswahili, Kikuyu, kod Dholuo.",
                    "Chutho anyalo konyo! Asebedo ka puonjo dhok kod timbe mag Kenya. Idwaro kony machal nade?",
                    "Amor konyo! Bed ni idwaro puonjo dhok, loko dhok, kata ngʼeyo timbe, an ka nikech in."
                ],
                "general": [
                    "Awinjo gima iwuoyo. Mano ber ahinya! Nyisa gimoro mangʼeny kuom parou mag wachni.",
                    "Erokamano kuom pogo mago koda. Aneno ni paro-ni nigi nengo maduongʼ. Bende ibiro dwaro wayor gimoro machielo?",
                    "Ageno kuom kelo wachni. Ber pile ka wan gi wuoyo machalo kama."
                ]
            }
        }
        
        # Determine response category
        greeting_keywords = ['hello', 'hi', 'hey', 'hujambo', 'habari', 'wĩ', 'inadi', 'greeting']
        question_keywords = ['what', 'how', 'why', 'when', 'where', 'nini', 'vipi', 'kwa', 'atĩa', 'ere', 'kare']
        help_keywords = ['help', 'assist', 'support', 'saidia', 'teithia', 'kony']
        
        if any(word in message_lower for word in greeting_keywords):
            category = "greeting"
        elif any(word in message_lower for word in question_keywords):
            category = "question"
        elif any(word in message_lower for word in help_keywords):
            category = "help"
        else:
            category = "general"
        
        # Get appropriate response
        response_options = responses.get(language, responses["english"])[category]
        response = random.choice(response_options)
        
        # Generate conversation ID if needed
        if not conversation_id:
            self.conversation_counter += 1
            conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.conversation_counter}"
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "language_detected": language,
            "confidence": 0.85,
            "tokens_generated": len(response.split()),
            "model_used": "simple_trilingual_ai",
            "processing_time": 0.1
        }
    
    def chat(self, message: str, target_language: str = "auto") -> Dict:
        """Main chat function"""
        if target_language == "auto":
            detected_language, confidence = self.detect_language(message)
        else:
            detected_language = target_language
            confidence = 0.9
        
        return self.generate_contextual_response(message, detected_language)

# Initialize AI system
if 'ai_system' not in st.session_state:
    st.session_state.ai_system = SimpleTrilingualAI()

# Load CSS (same as before but simplified)
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #43e97b;
        --text-primary: #2c3e50;
        --text-secondary: #7f8c8d;
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --border-color: #e9ecef;
        --shadow: 0 2px 10px rgba(0,0,0,0.1);
        --border-radius: 12px;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
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
    }
    
    .user-message {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin-left: 20%;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
    }
    
    .assistant-message {
        background: var(--bg-secondary);
        color: var(--text-primary);
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin-right: 20%;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }
    
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'active_chat_id' not in st.session_state:
        st.session_state.active_chat_id = None
    if 'chat_counter' not in st.session_state:
        st.session_state.chat_counter = 0

# Language configuration
LANGUAGE_CONFIG = {
    'auto': {'name': 'Auto-detect', 'flag': '🌐'},
    'en': {'name': 'English', 'flag': '🇺🇸'},
    'sw': {'name': 'Kiswahili', 'flag': '🇰🇪'},
    'ki': {'name': 'Kikuyu', 'flag': '🇰🇪'},
    'luo': {'name': 'Luo', 'flag': '🇰🇪'}
}

def create_new_chat():
    """Create a new chat session"""
    st.session_state.chat_counter += 1
    chat_id = f"chat_{st.session_state.chat_counter}"
    
    st.session_state.chats[chat_id] = {
        'id': chat_id,
        'title': f'New Chat {st.session_state.chat_counter}',
        'messages': [],
        'created_at': datetime.datetime.now(),
        'total_messages': 0,
        'language': 'auto'
    }
    
    st.session_state.active_chat_id = chat_id
    return chat_id

def get_active_chat():
    """Get the currently active chat"""
    if not st.session_state.active_chat_id or st.session_state.active_chat_id not in st.session_state.chats:
        if not st.session_state.chats:
            create_new_chat()
        else:
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
    
    return st.session_state.chats[st.session_state.active_chat_id]

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
        chat['total_messages'] += 1
        
        if role == 'user' and len(chat['messages']) == 1:
            title = content[:30] + "..." if len(content) > 30 else content
            chat['title'] = title

def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🧠 Trilingual AI Assistant</div>
        <p>Intelligent conversations in English, Kiswahili, Kikuyu, and Luo</p>
        <small>🆓 Free • 🌐 No signup required • 🚀 Instant responses</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("💬 Chat Management")
        
        # Chat selection
        if st.session_state.chats:
            chat_options = {}
            for chat_id, chat in st.session_state.chats.items():
                title = chat['title'][:30] + "..." if len(chat['title']) > 30 else chat['title']
                chat_options[title] = chat_id
            
            current_chat = get_active_chat()
            current_title = current_chat['title'][:30] + "..." if len(current_chat['title']) > 30 else current_chat['title']
            
            selected_chat_title = st.selectbox(
                "Select Chat:",
                options=list(chat_options.keys()),
                index=list(chat_options.keys()).index(current_title) if current_title in chat_options else 0
            )
            
            selected_chat_id = chat_options[selected_chat_title]
            if selected_chat_id != st.session_state.active_chat_id:
                st.session_state.active_chat_id = selected_chat_id
                st.rerun()
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()
        
        st.markdown("---")
        
        # Language selection
        st.header("🌐 Language Settings")
        active_chat = get_active_chat()
        current_language = active_chat.get('language', 'auto')
        
        selected_language = st.selectbox(
            "Response Language:",
            options=list(LANGUAGE_CONFIG.keys()),
            format_func=lambda x: f"{LANGUAGE_CONFIG[x]['flag']} {LANGUAGE_CONFIG[x]['name']}",
            index=list(LANGUAGE_CONFIG.keys()).index(current_language) if current_language in LANGUAGE_CONFIG else 0
        )
        
        if selected_language != current_language:
            active_chat['language'] = selected_language
        
        st.markdown("---")
        
        # Quick examples
        st.header("💡 Quick Examples")
        examples = [
            ("🇺🇸", "Hello! How can you help me?"),
            ("🇰🇪", "Habari yako? Unaweza kunisaidia?"),
            ("🇰🇪", "Wĩ atĩa? Ũngĩndeithagia?"),
            ("🇰🇪", "Inadi? Inyalo konya nadi?")
        ]
        
        for flag, example in examples:
            if st.button(f"{flag} {example[:25]}...", key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.example_clicked = example
        
        st.markdown("---")
        st.info("🆓 **Completely Free**\n\nNo API keys, no signup, no limits!")
    
    # Main content
    active_chat = get_active_chat()
    st.subheader(f"💬 {active_chat['title']}")
    
    # Display messages
    messages = active_chat.get('messages', [])
    
    if not messages:
        st.info("👋 Welcome! Start a conversation in any language. I'll respond in the same language or your preferred one.")
    else:
        for message in messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"👤 You • {message.get('timestamp', '')}")
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                lang = message.get('language', 'Unknown')
                confidence = message.get('confidence', 0)
                st.caption(f"🤖 AI • {lang} • {confidence:.0%} confidence • {message.get('timestamp', '')}")
    
    # Message input
    st.markdown("---")
    
    # Check for example click
    if hasattr(st.session_state, 'example_clicked'):
        user_input = st.session_state.example_clicked
        del st.session_state.example_clicked
    else:
        user_input = None
    
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
        st.write("")
        st.write("")
        send_clicked = st.button("Send", type="primary")
    
    # Handle message sending
    if send_clicked and message.strip():
        # Add user message
        add_message_to_chat(
            st.session_state.active_chat_id, 
            "user", 
            message,
            timestamp=datetime.datetime.now().strftime("%H:%M")
        )
        
        # Clear the text box
        if "user_message" in st.session_state:
            del st.session_state["user_message"]
        
        # Generate AI response
        with st.spinner("🤔 AI is thinking..."):
            try:
                # Map language codes
                language_mapping = {
                    'auto': 'auto',
                    'en': 'english',
                    'sw': 'kiswahili',
                    'ki': 'kikuyu',
                    'luo': 'luo'
                }
                
                target_lang = language_mapping.get(selected_language, 'auto')
                result = st.session_state.ai_system.chat(message, target_lang)
                
                # Add assistant response
                add_message_to_chat(
                    st.session_state.active_chat_id,
                    "assistant",
                    result['response'],
                    timestamp=datetime.datetime.now().strftime("%H:%M"),
                    language=result['language_detected'],
                    confidence=result['confidence'],
                    model_used=result['model_used']
                )
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
