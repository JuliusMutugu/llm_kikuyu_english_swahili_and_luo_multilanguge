"""
Enhanced Conversational System with Better Language Support
"""

import random
import datetime
from typing import Dict, List, Optional
import json

class EnhancedConversationalSystem:
    """Enhanced conversation system with proper language support and context"""
    
    def __init__(self):
        self.conversation_history = {}  # Store by conversation_id
        self.language_templates = self._load_language_templates()
        self.context_patterns = self._load_context_patterns()
        
    def _load_language_templates(self) -> Dict:
        """Load comprehensive language templates for better conversations"""
        return {
            "english": {
                "greetings": {
                    "responses": [
                        "Hello! How can I help you today?",
                        "Hi there! What would you like to talk about?",
                        "Good day! I'm here to assist you with anything you need.",
                        "Welcome! How are you doing today?",
                        "Hey! Great to see you. What's on your mind?"
                    ],
                    "follow_ups": [
                        "Is there something specific you'd like help with?",
                        "What brings you here today?",
                        "How has your day been so far?"
                    ]
                },
                "questions": {
                    "responses": [
                        "That's a great question! Let me help you with that.",
                        "Interesting question! Here's what I think:",
                        "I'd be happy to help you understand that better.",
                        "Good point! Let me explain that for you.",
                        "That's something I can definitely help with."
                    ],
                    "explanations": [
                        "From what I understand, ",
                        "Based on my knowledge, ",
                        "Here's how I see it: ",
                        "Let me break this down for you: ",
                        "The way I understand it is: "
                    ]
                },
                "general": {
                    "responses": [
                        "I understand what you're saying.",
                        "That makes sense to me.",
                        "I can see your point of view.",
                        "That's very interesting!",
                        "Thanks for sharing that with me."
                    ],
                    "continuations": [
                        "Would you like to tell me more about that?",
                        "How do you feel about that?",
                        "What do you think about it?",
                        "Is there more you'd like to discuss?"
                    ]
                },
                "help": {
                    "responses": [
                        "I'm here to help! What do you need assistance with?",
                        "Of course! I'd be happy to help you.",
                        "Let me see how I can assist you with that.",
                        "I'm ready to help! What would you like to know?",
                        "Absolutely! How can I support you today?"
                    ]
                },
                "farewells": {
                    "responses": [
                        "Thank you for the great conversation!",
                        "It was nice talking with you!",
                        "Have a wonderful day!",
                        "Take care and feel free to come back anytime!",
                        "Goodbye! Hope to chat with you again soon!"
                    ]
                }
            },
            "kiswahili": {
                "greetings": {
                    "responses": [
                        "Habari! Naweza kukusaidia vipi leo?",
                        "Hujambo! Ni nini ungependa tuzungumze?",
                        "Karibu sana! Nipo hapa kukusaidia.",
                        "Salamu! Hujambo, unahali aje?",
                        "Ahsante kuja! Ni nini kinachokusumbua?"
                    ],
                    "follow_ups": [
                        "Je, kuna kitu fulani unachotaka msaada?",
                        "Ni nini kinakuleta hapa leo?",
                        "Umesikaje leo?"
                    ]
                },
                "questions": {
                    "responses": [
                        "Hilo ni swali zuri sana! Hebu nikusaidie.",
                        "Swali la kuvutia! Hiki ndicho nafikiri:",
                        "Nitafurahi kukusaidia kuelewa hilo vizuri.",
                        "Umba mzuri! Hebu nikuleleze:",
                        "Hilo ni jambo ninaweza kukusaidia nalo."
                    ],
                    "explanations": [
                        "Kwa vile ninavyoelewa, ",
                        "Kulingana na ujuzi wangu, ",
                        "Ninaona hivi: ",
                        "Hebu nikuleleze kwa ufupi: ",
                        "Ninaelewa hivi: "
                    ]
                },
                "general": {
                    "responses": [
                        "Nimeelewa unachosema.",
                        "Hilo lina maana kwangu.",
                        "Ninaweza kuona mtazamo wako.",
                        "Hilo ni jambo la kuvutia sana!",
                        "Asante kwa kunishirikisha hilo."
                    ],
                    "continuations": [
                        "Je, ungependa kuniambia zaidi kuhusu hilo?",
                        "Unahisije kuhusu hilo?",
                        "Unafikiri nini kuhusu hilo?",
                        "Kuna zaidi ungependa tujadili?"
                    ]
                },
                "help": {
                    "responses": [
                        "Nipo hapa kukusaidia! Unahitaji msaada gani?",
                        "Bila shaka! Nitafurahi kukusaidia.",
                        "Hebu nione jinsi ninaweza kukusaidia.",
                        "Niko tayari kusaidia! Unataka kujua nini?",
                        "Kabisa! Ninaweza kukuunga mkono vipi leo?"
                    ]
                },
                "farewells": {
                    "responses": [
                        "Asante kwa mazungumzo mazuri!",
                        "Ilikuwa ni furaha kuzungumza nawe!",
                        "Uwe na siku njema!",
                        "Jali nafsi yako na ujisikie huru kurudi wakati wowote!",
                        "Kwaheri! Natumai tutazungumza tena!"
                    ]
                }
            },
            "kikuyu": {
                "greetings": {
                    "responses": [
                        "Wĩ mwega! Nĩngũkũteithia atĩa ũmũthĩ?",
                        "Wĩ atĩa! Nĩ kĩĩ ũngĩenda tũkwarie?",
                        "Wĩ wega mũno! Ndĩ gũkũ gũgũteithia.",
                        "Njĩra! Wĩ atĩa, ũrĩ atĩa?",
                        "Ngũkena gũkũona! Nĩ kĩĩ kĩragũthĩĩnia?"
                    ],
                    "follow_ups": [
                        "Nĩ harĩ ũndũ ũmwe ũngĩenda ũteithio?",
                        "Nĩ kĩĩ kĩrakũrehe gũkũ ũmũthĩ?",
                        "Ũrĩ atĩa ũmũthĩ?"
                    ]
                },
                "questions": {
                    "responses": [
                        "Ũcio nĩ mũũria mwega mũno! Reke ngũteithie.",
                        "Mũũria wa kũgegania! Ũyũ nĩguo meciiria makwa:",
                        "Nĩngũkena gũgũteithia gũtaũkĩrwo nĩ ũguo wega.",
                        "Meciiria mega! Reke ngũhe ũhoro:",
                        "Ũcio nĩ ũndũ ingĩgũteithia naguo."
                    ],
                    "explanations": [
                        "Kũringana na ũrĩa ndĩmenyaga, ",
                        "Kũringana na ũmenyo wakwa, ",
                        "Nĩnjĩonaga atĩrĩ: ",
                        "Reke ngũtaarĩrie na ũhotu: ",
                        "Nĩndĩmenyaga atĩrĩ: "
                    ]
                },
                "general": {
                    "responses": [
                        "Nĩndĩmenya ũrĩa ũkuuga.",
                        "Ũguo nĩguo kũgĩrĩire harĩ niĩ.",
                        "Nĩngũona mũno kĩrĩa ũrona.",
                        "Ũcio nĩ ũndũ wa kũgegania mũno!",
                        "Nĩngũkena nĩ ũndũ wa kũnjĩra ũguo."
                    ],
                    "continuations": [
                        "Ũngĩenda kũnjĩra ũhoro ũngĩ ũkoniĩ ũguo?",
                        "Wĩ atĩa ũkoniĩ ũguo?",
                        "Ũreciiria atĩa ũkoniĩ ũguo?",
                        "Nĩ harĩ ũndũ ũngĩ ũngĩenda tũkwarie?"
                    ]
                },
                "help": {
                    "responses": [
                        "Ndĩ gũkũ gũgũteithia! Ũrenda ũteithio ũrĩkũ?",
                        "Ti-itherũ! Nĩngũkena gũgũteithia.",
                        "Reke njone njĩra ya gũgũteithia.",
                        "Ndĩ mũthĩndĩ wa gũteithia! Ũrenda kũmenya atĩa?",
                        "Tiguo! Ingĩgũteithia atĩa ũmũthĩ?"
                    ]
                },
                "farewells": {
                    "responses": [
                        "Nĩngũkena nĩ ũndũ wa kwaria mega!",
                        "Nĩkwagĩire mũno gũkwaria nawe!",
                        "Gĩa na mũthenya mwega!",
                        "Wĩmenyerere na ũcooke rĩrĩa rĩothe ũrenda!",
                        "Tigũo! Ndĩrenda tũkwarie rĩngĩ!"
                    ]
                }
            },
            "luo": {
                "greetings": {
                    "responses": [
                        "Inadi maber! Ere kaka ma anyalo konyigo kawuono?",
                        "Amosi maber! En angʼo ma dwamoyweyogo?",
                        "Oiore maber! An ka mondo akonyi.",
                        "Aora! Idhi nade, in gi chuny mariambo?",
                        "Amor kuom biro! En angʼo ma chando iparinyi?"
                    ],
                    "follow_ups": [
                        "Nitie gimoro ka ma idwaro mondo akonyi kuome?",
                        "En angʼo ma okelogi ka kawuono?",
                        "Chunyi nade kawuono?"
                    ]
                },
                "questions": {
                    "responses": [
                        "Mano en penjo maber ahinya! We mondo akonyi.",
                        "Penjo ma konyo chuny! Ma e gima aparo:",
                        "Abiro bedo mamor kuom konyogi mondo inyis gima tiend.",
                        "Paro maber! We mondo anyisi:",
                        "Mano en gima anyalo konyigo kuome."
                    ],
                    "explanations": [
                        "Kaluwore gi gima angʼeyo, ",
                        "Kaluwore gi rieko ma an-go, ",
                        "Aneno kama: ",
                        "We mondo anyisi gi yore mayot: ",
                        "Angʼeyo ni: "
                    ]
                },
                "general": {
                    "responses": [
                        "Awinjo gima iwachono.",
                        "Mano nigi tiende mana kuoma.",
                        "Anyalo neno kaka ineno.",
                        "Mano en gimoro ma nyiso rieko ahinya!",
                        "Erokamano kuom pogo mano koda."
                    ],
                    "continuations": [
                        "Dibed ni idwaro nyisa gimoro machielo kuom mano?",
                        "Chunyi nade kuom mano?",
                        "Iparo nadi kuom mano?",
                        "Nitie gimoro machielo ma dwamowacho?"
                    ]
                },
                "help": {
                    "responses": [
                        "An ka mondo akonyi! Idwaro kony mane?",
                        "Adiera! Abiro bedo mamor kuom konyigi.",
                        "We mondo ane kaka anyalo konyigi.",
                        "An gi ikruok mar konyo! Idwaro ngʼeyo angʼo?",
                        "Kamano! Anyalo konyigo kaka nadi kawuono?"
                    ]
                },
                "farewells": {
                    "responses": [
                        "Erokamano kuom weche mabernigi!",
                        "Ne ber ahinya wuoyo kodi!",
                        "Bed gi odiechieng maber!",
                        "Rit chunyi maber kendo duog sa asaya ma dwaro!",
                        "Oriti! Agombo ni wanawacho kendo!"
                    ]
                }
            }
        }
    
    def _load_context_patterns(self) -> Dict:
        """Load patterns for context understanding"""
        return {
            "greeting_patterns": [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "habari", "hujambo", "salamu", "mambo", "vipi",
                "wĩ", "atĩa", "wĩ mwega", "wĩ wega",
                "inadi", "amosi", "aora", "ere"
            ],
            "question_patterns": [
                "what", "how", "why", "when", "where", "who", "which",
                "nini", "vipi", "kwa nini", "lini", "wapi", "nani", "gani",
                "atĩa", "ngai", "nĩ kĩĩ", "rĩ", "ha", "nũũ", "ũrĩkũ",
                "angʼo", "kaka", "ere", "ma", "ka", "ng", "mane"
            ],
            "help_patterns": [
                "help", "assist", "support", "need", "can you",
                "msaada", "saidia", "unganisha", "unga mkono", "unaweza",
                "teithia", "gũteithia", "ũngĩ", "gũtuĩka", "ũngĩ",
                "kony", "kony", "konyo", "nyalo", "inyalo"
            ],
            "farewell_patterns": [
                "bye", "goodbye", "see you", "farewell", "take care",
                "kwaheri", "tutaonana", "uende salama", "jali nafsi",
                "tiguo", "ona", "gĩa wega", "wĩmenyerere",
                "oriti", "wanaonore", "bed gi"
            ]
        }
    
    def _detect_language(self, text: str) -> tuple:
        """Enhanced language detection"""
        text_lower = text.lower()
        
        # Score for each language
        scores = {"english": 0, "kiswahili": 0, "kikuyu": 0, "luo": 0}
        
        # Kikuyu indicators
        kikuyu_indicators = ['wĩ', 'atĩa', 'ũrĩ', 'nĩ', 'ũ', 'ĩ', 'gĩkũyũ', 'njĩra', 'mwega', 'teithia', 'ũngĩ']
        for indicator in kikuyu_indicators:
            if indicator in text_lower:
                scores["kikuyu"] += 2
        
        # Luo indicators
        luo_indicators = ['inadi', 'adhi', 'maber', 'gima', 'inyalo', 'konya', 'ere', 'dhok', 'amosi', 'chuny']
        for indicator in luo_indicators:
            if indicator in text_lower:
                scores["luo"] += 2
        
        # Kiswahili indicators
        swahili_indicators = ['habari', 'mzuri', 'asante', 'karibu', 'pole', 'haraka', 'lugha', 'sana', 'hujambo', 'mambo']
        for indicator in swahili_indicators:
            if indicator in text_lower:
                scores["kiswahili"] += 2
        
        # English gets default score
        scores["english"] = 1
        
        # Find the language with highest score
        detected_lang = max(scores.items(), key=lambda x: x[1])
        confidence = min(detected_lang[1] / 3.0, 1.0)  # Normalize confidence
        
        return detected_lang[0], confidence
    
    def _identify_intent(self, text: str) -> str:
        """Identify the intent of the message"""
        text_lower = text.lower()
        
        # Check for greeting patterns
        for pattern in self.context_patterns["greeting_patterns"]:
            if pattern in text_lower:
                return "greetings"
        
        # Check for question patterns
        for pattern in self.context_patterns["question_patterns"]:
            if pattern in text_lower:
                return "questions"
        
        # Check for help patterns
        for pattern in self.context_patterns["help_patterns"]:
            if pattern in text_lower:
                return "help"
        
        # Check for farewell patterns
        for pattern in self.context_patterns["farewell_patterns"]:
            if pattern in text_lower:
                return "farewells"
        
        return "general"
    
    def generate_response(
        self,
        message: str,
        language: str = "auto",
        conversation_id: str = None,
        max_length: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict:
        """Generate contextual conversational response"""
        
        start_time = datetime.datetime.now()
        
        # Auto-detect language if needed
        if language == "auto":
            detected_lang, confidence = self._detect_language(message)
        else:
            detected_lang, confidence = language, 0.9
        
        # Generate conversation ID if needed
        if not conversation_id:
            conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize conversation history for this ID
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        # Add user message to history
        self.conversation_history[conversation_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Identify intent
        intent = self._identify_intent(message)
        
        # Generate response based on language and intent
        try:
            templates = self.language_templates.get(detected_lang, self.language_templates["english"])
            intent_templates = templates.get(intent, templates["general"])
            
            # Choose appropriate response
            if "responses" in intent_templates:
                base_response = random.choice(intent_templates["responses"])
            else:
                base_response = "I understand what you're saying."
            
            # Add context-aware follow-up
            conversation_length = len(self.conversation_history[conversation_id])
            
            if conversation_length > 1:
                # For ongoing conversations, add more contextual responses
                if intent == "questions" and "explanations" in intent_templates:
                    explanation_start = random.choice(intent_templates["explanations"])
                    base_response = explanation_start + base_response.lower()
                elif intent == "general" and "continuations" in intent_templates:
                    if random.random() < 0.3:  # 30% chance to add follow-up
                        follow_up = random.choice(intent_templates["continuations"])
                        base_response += " " + follow_up
            else:
                # For first interaction, add welcoming follow-up
                if intent == "greetings" and "follow_ups" in intent_templates:
                    if random.random() < 0.5:  # 50% chance
                        follow_up = random.choice(intent_templates["follow_ups"])
                        base_response += " " + follow_up
            
            # Add user's name if mentioned in conversation
            for msg in self.conversation_history[conversation_id]:
                if "my name is" in msg["content"].lower() or "i am" in msg["content"].lower():
                    # Extract potential name and use it occasionally
                    pass
            
            # Add message to conversation history
            self.conversation_history[conversation_id].append({
                "role": "assistant",
                "content": base_response,
                "timestamp": datetime.datetime.now().isoformat(),
                "intent": intent,
                "language": detected_lang
            })
            
        except Exception as e:
            base_response = "I'm here to help! Could you please rephrase that?"
            detected_lang = "english"
            confidence = 0.5
        
        # Calculate processing time
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        
        return {
            "response": base_response,
            "conversation_id": conversation_id,
            "language_detected": detected_lang,
            "confidence": confidence,
            "tokens_generated": len(base_response.split()),
            "model_used": "enhanced_conversational_system",
            "expert_usage": None,
            "processing_time": processing_time,
            "intent": intent,
            "conversation_turn": len(self.conversation_history[conversation_id])
        }

# Create global instance
enhanced_conversation_system = EnhancedConversationalSystem()
