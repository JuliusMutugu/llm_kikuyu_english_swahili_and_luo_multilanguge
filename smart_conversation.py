"""
Intelligent Conversation Engine
Provides meaningful responses in multiple languages
"""

import random
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class ConversationEngine:
    """Advanced conversation engine with context awareness"""
    
    def __init__(self):
        self.conversation_history = {}
        self.language_patterns = {
            'english': {
                'greetings': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
                'questions': ['what', 'how', 'when', 'where', 'why', 'who', 'can you', 'do you'],
                'thanks': ['thank', 'thanks', 'appreciate'],
                'goodbye': ['bye', 'goodbye', 'see you', 'farewell']
            },
            'kiswahili': {
                'greetings': ['habari', 'hujambo', 'mambo', 'salamu'],
                'questions': ['nini', 'vipi', 'wapi', 'lini', 'kwa nini', 'nani', 'unaweza'],
                'thanks': ['asante', 'shukrani'],
                'goodbye': ['kwaheri', 'tutaonana', 'kwa heri']
            },
            'kikuyu': {
                'greetings': ['wĩ', 'atĩa', 'wĩ mwega', 'wĩ atĩa'],
                'questions': ['atĩa', 'ũtĩ', 'kũ', 'rĩ', 'nĩkĩ', 'nũũ', 'ũngĩ'],
                'thanks': ['nĩngũteithagia', 'asante'],
                'goodbye': ['tiguo', 'tuonane']
            },
            'luo': {
                'greetings': ['inadi', 'amosi', 'ber', 'oyawore'],
                'questions': ['angʼo', 'kaka', 'kanye', 'karangʼo', 'mano', 'ngʼa', 'inyalo'],
                'thanks': ['erokamano', 'asante'],
                'goodbye': ['oriti', 'wanene']
            }
        }
        
        self.responses = {
            'english': {
                'greetings': [
                    "Hello! I'm delighted to chat with you today. How can I assist you?",
                    "Hi there! Welcome to our multilingual conversation. What would you like to talk about?",
                    "Good day! I'm here to help you with anything you need. How are you doing?",
                    "Hello! It's wonderful to meet you. I can communicate in English, Kiswahili, Kikuyu, and Luo. What interests you?"
                ],
                'questions': [
                    "That's an excellent question! Let me think about that for you.",
                    "I'd be happy to help you with that. Here's what I know:",
                    "Great question! I can certainly help you understand this better.",
                    "That's very interesting! Let me provide you with some insight."
                ],
                'capabilities': [
                    "I'm a multilingual AI assistant that can communicate in English, Kiswahili, Kikuyu, and Luo. I can help with conversations, answer questions, and provide information in any of these languages.",
                    "I specialize in multilingual communication across East African languages. I can help with translations, cultural context, and general conversations.",
                    "My main strength is facilitating natural conversations in multiple languages while understanding cultural nuances."
                ],
                'general': [
                    "I understand what you're saying. Please tell me more about what you'd like to discuss.",
                    "That's interesting! I'd love to continue this conversation with you.",
                    "Thank you for sharing that with me. What else would you like to talk about?",
                    "I appreciate you taking the time to chat with me. How else can I help you today?"
                ],
                'learning': [
                    "I'm always learning from our conversations. Each interaction helps me understand better.",
                    "Learning languages and cultures is fascinating! I enjoy every conversation we have.",
                    "Your questions help me become a better conversational partner. Thank you!"
                ]
            },
            'kiswahili': {
                'greetings': [
                    "Habari yako! Nimefurahi sana kukutana nawe leo. Naweza kukusaidiaje?",
                    "Hujambo! Karibu kwenye mazungumzo yetu. Tungependa kuzungumza kuhusu nini?",
                    "Habari za leo! Niko hapa kukusaidia kwa chochote. Unahali aje?",
                    "Salamu! Ni furaha kukutana nawe. Naweza kuzungumza Kiingereza, Kiswahili, Kikuyu, na Kiluo."
                ],
                'questions': [
                    "Hilo ni swali zuri sana! Niruhusu nifikiirie kuhusu hilo.",
                    "Ninafurahi kukusaidia na hilo. Hiki ndicho ninachojua:",
                    "Swali zuri! Hakika nitakusaidia kuelewa hili vizuri zaidi.",
                    "Hilo ni la kuvutia sana! Niruhusu nikupe maelezo."
                ],
                'capabilities': [
                    "Mimi ni msaidizi wa AI anayeweza kuzungumza lugha nyingi - Kiingereza, Kiswahili, Kikuyu, na Kiluo. Naweza kusaidia na mazungumzo, majibu ya maswali, na maelezo katika lugha hizi zote.",
                    "Nimejifunza lugha za Afrika Mashariki. Naweza kusaidia na tafsiri, utamaduni, na mazungumzo ya kawaida.",
                    "Uongozi wangu mkuu ni kuwezesha mazungumzo ya asili katika lugha nyingi huku nikielewa utamaduni."
                ],
                'general': [
                    "Naelewa unachosema. Tafadhali niambie zaidi kuhusu kile ungependa kujadili.",
                    "Hilo ni la kuvutia! Ningependa kuendelea na mazungumzo haya nawe.",
                    "Asante kwa kunishirikisha hilo. Kitu gani kingine ungependa kuzungumza?",
                    "Nasukuru kwa muda wako wa kuzungumza nami. Ninawezaje kukusaidia zaidi leo?"
                ]
            },
            'kikuyu': {
                'greetings': [
                    "Wĩ mwega! Nĩngũkenete mũno gũgũcemania ũmũthĩ. Ingĩkũteithia atĩa?",
                    "Wĩ atĩa! Wamũkĩra gũkũ kũrĩa tũciaragia. Tũkwenda kwaria ũhoro wa atĩa?",
                    "Wĩ mwega wa ũmũthĩ! Ndĩ gũkũ gũgũteithia harĩ ũndũ o wothe. Wĩ atĩa?",
                    "Ndamiũkai! Nĩ gĩkeno gũgũcemania. Nĩngũhota kwaria Gĩthũngũ, Kiswahili, Gĩkũyũ, na Kiluo."
                ],
                'questions': [
                    "Ũcio nĩ mũũria mwega mũno! Reke ndĩciirie ũhoro ũcio.",
                    "Nĩngũkena gũgũteithia na ũndũ ũcio. Ũyũ nĩguo ũũrũ ũrĩa njũũĩ:",
                    "Mũũria mwega! Ti-itherũ nĩngũgũteithia gũtaũkĩrwo nĩ ũndũ ũcio wega.",
                    "Ũndũ ũcio nĩ wa kũgegania mũno! Reke ngũhe ũhoro."
                ],
                'capabilities': [
                    "Niĩ ndĩ mũteithĩrĩria wa AI ũngĩaria rũgano rũingĩ - Gĩthũngũ, Kiswahili, Gĩkũyũ, na Kiluo. Nĩngũhota gũteithia na gũthiĩ, macookio ma ciũria, na ũhoro kũringana na rũgano rũrũ ruothe.",
                    "Njigĩte rũgano rwa Afrika ya Ithũngũ. Nĩngũhota gũteithia na gutaũra, ngero, na mĩĩgano ya wega.",
                    "Ũndũ wakwa mũnene nĩ kũnyiihia mĩĩgano ya ũũru thĩinĩ wa rũgano ruingĩ ngiuga ngero."
                ],
                'general': [
                    "Nĩndamenya ũrĩa ũroiga. Tafadhali njĩra maũndũ mangĩ makoniĩ ũrĩa ũngwenda kwaria.",
                    "Ũndũ ũcio nĩ wa kũgegania! Nĩngwenda tũthiĩ na mbere na mĩĩgano ĩno nawe.",
                    "Nĩngũkenagia nĩ ũndũ wa kũnjĩra ũguo. Nĩ kĩĩ kĩngĩ ũngwenda kwaria?",
                    "Nĩndakenagia nĩ ũndũ wa ihinda rĩaku rĩa kwaria na niĩ. Ingĩgũteithia atĩa ingĩ ũmũthĩ?"
                ]
            },
            'luo': {
                'greetings': [
                    "Inadi! Amor maduongʼ kuom romoga kawuono. Ere kaka anyalo konyogi?",
                    "Amosi! Winjuru kar wuoyo marwa. Wawuoyogo kuom angʼo?",
                    "Inadi maber mar kawuono! An ka mondo akonyi kuom gimoro amora. Idhi nade?",
                    "Ojwok! En mor kuom romogi. Anyalo wuoyo gi Dholuo, Kiswahili, Kikuyu, gi Dhoingereza."
                ],
                'questions': [
                    "Mano en penjo maber maduongʼ! We mondo aparogi kuom mano.",
                    "Amor kuom konyogi gi mano. Ma e gima angʼeyo:",
                    "Penjo maber! Adieri abiro konyogi winjo mani maber.",
                    "Mano lich ahinya! We mondo amii weche."
                ],
                'capabilities': [
                    "An jakony mar AI manyalo wuoyo gi dhok mangʼeny - Dholuo, Kiswahili, Kikuyu, gi Dhoingereza. Anyalo konyo gi wuoyo, dwoko penjo, gi weche e dhok misegogi.",
                    "Asepuonjora dhok mag Africa mar Wuok Chiengʼ. Anyalo konyo gi lokruok, sigungu, gi wuoyo mapoth.",
                    "Tich maduongʼ mara en miyo wuoyo kik bed matek e dhok mongʼeny ka angʼeyo sigungu."
                ],
                'general': [
                    "Awinjo gima iwacho. Kiyie inyisa mangʼeny kuom gima idwaro wuoyogo.",
                    "Mano lich! Adwaro mondo wadhi nyime gi wuokonigi.",
                    "Erokamano kuom pogo mano koda. En angʼo machielo midwaro wuoyogo?",
                    "Apako seche mago mag wuoyo koda. Ere kaka anyalo konyogi kawuono?"
                ]
            }
        }

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the language of input text"""
        text_lower = text.lower()
        scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = 0
            total_words = len(text_lower.split())
            
            for category, words in patterns.items():
                for word in words:
                    if word in text_lower:
                        score += 2  # Higher weight for exact matches
                        
            # Check for language-specific characters
            if lang == 'kikuyu' and any(char in text for char in 'ĩũẽĩũ'):
                score += 3
            elif lang == 'luo' and any(char in text for char in 'ʼ'):
                score += 3
                
            scores[lang] = score / max(total_words, 1)
        
        # Default to English if no clear detection
        if not scores or max(scores.values()) < 0.3:
            return 'english', 0.6
            
        best_lang = max(scores, key=scores.get)
        confidence = min(scores[best_lang], 1.0)
        
        return best_lang, confidence

    def categorize_input(self, text: str, language: str) -> str:
        """Categorize the input to determine response type"""
        text_lower = text.lower()
        
        if not language in self.language_patterns:
            language = 'english'
            
        patterns = self.language_patterns[language]
        
        # Check for greetings
        if any(word in text_lower for word in patterns['greetings']):
            return 'greetings'
            
        # Check for questions
        if any(word in text_lower for word in patterns['questions']) or '?' in text:
            return 'questions'
            
        # Check for thanks
        if any(word in text_lower for word in patterns['thanks']):
            return 'thanks'
            
        # Check for goodbye
        if any(word in text_lower for word in patterns['goodbye']):
            return 'goodbye'
            
        # Check for capability questions
        capability_keywords = ['what can you', 'what do you', 'tell me about', 'about you', 'capabilities', 'uwezo', 'ũhoti', 'tekoni']
        if any(keyword in text_lower for keyword in capability_keywords):
            return 'capabilities'
            
        # Check for learning/AI related
        learning_keywords = ['learn', 'ai', 'artificial', 'robot', 'machine', 'jifunza', 'ũthomi', 'puonjore']
        if any(keyword in text_lower for keyword in learning_keywords):
            return 'learning'
            
        return 'general'

    def generate_contextual_response(self, text: str, conversation_id: str = None) -> Dict:
        """Generate a contextual response based on input"""
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"smart_conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Detect language
        detected_lang, confidence = self.detect_language(text)
        
        # Categorize input
        category = self.categorize_input(text, detected_lang)
        
        # Get appropriate responses
        if detected_lang in self.responses and category in self.responses[detected_lang]:
            possible_responses = self.responses[detected_lang][category]
        else:
            # Fallback to English general responses
            possible_responses = self.responses['english']['general']
            detected_lang = 'english'
        
        # Select response (can be made smarter with context)
        response = random.choice(possible_responses)
        
        # Add some context-specific modifications
        if 'name' in text.lower():
            if detected_lang == 'english':
                response += " I'm your multilingual AI assistant, and I'm here whenever you need help!"
            elif detected_lang == 'kiswahili':
                response += " Mimi ni msaidizi wako wa AI anayeongea lugha nyingi, niko hapa wakati wowote unahitaji msaada!"
            elif detected_lang == 'kikuyu':
                response += " Niĩ ndĩ mũteithĩrĩria waku wa AI ũrĩa ũaraaga rũgano ruingĩ, ndĩ gũkũ rĩrĩa rĩothe ũkabataro ũteithio!"
            elif detected_lang == 'luo':
                response += " An jakonygi mar AI mawuoyo gi dhok mongʼeny, an ka sa asaya ma idwaro kony!"
        
        # Store conversation context (for future improvements)
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        self.conversation_history[conversation_id].append({
            'input': text,
            'response': response,
            'language': detected_lang,
            'category': category,
            'timestamp': datetime.now()
        })
        
        return {
            'response': response,
            'language_detected': detected_lang,
            'confidence': confidence,
            'category': category,
            'tokens_generated': len(response.split()),
            'conversation_id': conversation_id
        }

# Global instance
conversation_engine = ConversationEngine()

def generate_smart_response(message: str, conversation_id: str = None) -> Dict:
    """Main function to generate intelligent responses"""
    return conversation_engine.generate_contextual_response(message, conversation_id)

if __name__ == "__main__":
    # Test the conversation engine
    test_messages = [
        "Hello, how are you?",
        "Habari yako?",
        "Wĩ atĩa?",
        "Inadi?",
        "What can you do?",
        "Unaweza nini?",
        "Tell me about yourself",
        "Thank you very much"
    ]
    
    print("Testing Conversation Engine:")
    print("="*50)
    
    for msg in test_messages:
        result = generate_smart_response(msg, "test_conversation")
        print(f"Input: {msg}")
        print(f"Language: {result['language_detected']} ({result['confidence']:.2f} confidence)")
        print(f"Response: {result['response']}")
        print(f"Category: {result['category']}")
        print("-"*50)
