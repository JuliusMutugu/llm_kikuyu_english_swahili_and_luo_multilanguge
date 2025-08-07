"""
Continuous Learning System
Learns from web sources and user conversations to improve responses
"""

import requests
import json
import re
import datetime
import sqlite3
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import feedparser
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationLearner:
    """Learns from user conversations to improve responses"""
    
    def __init__(self, db_path="conversation_learning.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            user_message TEXT,
            ai_response TEXT,
            language TEXT,
            confidence REAL,
            user_rating INTEGER,
            timestamp DATETIME,
            learned_patterns TEXT
        )
        ''')
        
        # Common patterns table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learned_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT,
            pattern TEXT,
            language TEXT,
            frequency INTEGER DEFAULT 1,
            success_rate REAL DEFAULT 0.0,
            last_used DATETIME,
            created_at DATETIME
        )
        ''')
        
        # Knowledge base table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            content TEXT,
            source TEXT,
            language TEXT,
            relevance_score REAL,
            last_updated DATETIME,
            hash_key TEXT UNIQUE
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_conversation(self, conversation_id: str, user_message: str, 
                          ai_response: str, language: str, confidence: float):
        """Record a conversation for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract patterns from user message
        patterns = self.extract_patterns(user_message, language)
        
        cursor.execute('''
        INSERT INTO conversations 
        (conversation_id, user_message, ai_response, language, confidence, timestamp, learned_patterns)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (conversation_id, user_message, ai_response, language, confidence, 
              datetime.datetime.now(), json.dumps(patterns)))
        
        # Update pattern frequency
        for pattern in patterns:
            self.update_pattern_frequency(pattern, language)
        
        conn.commit()
        conn.close()
        
    def extract_patterns(self, message: str, language: str) -> List[str]:
        """Extract common patterns from user messages"""
        patterns = []
        message_lower = message.lower()
        
        # Common question patterns
        question_patterns = {
            'english': [
                'what is', 'how to', 'can you', 'tell me about', 'explain', 
                'why is', 'where is', 'when is', 'who is'
            ],
            'kiswahili': [
                'ni nini', 'vipi', 'unaweza', 'niambie', 'eleza',
                'kwa nini', 'wapi', 'lini', 'nani'
            ],
            'kikuyu': [
                'nĩ atĩa', 'ũtĩ', 'ũngĩ', 'njĩra', 'menya',
                'nĩkĩ', 'kũ', 'rĩ', 'nũũ'
            ],
            'luo': [
                'en angʼo', 'kaka', 'inyalo', 'nyisa', 'lero',
                'mano', 'kanye', 'karangʼo', 'ngʼa'
            ]
        }
        
        # Check for patterns
        lang_patterns = question_patterns.get(language, question_patterns['english'])
        for pattern in lang_patterns:
            if pattern in message_lower:
                patterns.append(f"question_pattern_{pattern.replace(' ', '_')}")
        
        # Extract entities (simple approach)
        if any(word in message_lower for word in ['name', 'jina', 'rĩtwa', 'nying']):
            patterns.append('asking_name')
        
        if any(word in message_lower for word in ['help', 'msaada', 'ũteithio', 'kony']):
            patterns.append('requesting_help')
            
        return patterns
    
    def update_pattern_frequency(self, pattern: str, language: str):
        """Update pattern frequency in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO learned_patterns 
        (pattern_type, pattern, language, frequency, last_used, created_at)
        VALUES (
            'conversation_pattern', ?, ?, 
            COALESCE((SELECT frequency FROM learned_patterns 
                     WHERE pattern = ? AND language = ?) + 1, 1),
            ?, 
            COALESCE((SELECT created_at FROM learned_patterns 
                     WHERE pattern = ? AND language = ?), ?)
        )
        ''', (pattern, language, pattern, language, datetime.datetime.now(),
              pattern, language, datetime.datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_response_suggestions(self, user_message: str, language: str) -> List[str]:
        """Get response suggestions based on learned patterns"""
        patterns = self.extract_patterns(user_message, language)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        suggestions = []
        for pattern in patterns:
            cursor.execute('''
            SELECT ai_response, confidence FROM conversations 
            WHERE learned_patterns LIKE ? AND language = ? 
            ORDER BY confidence DESC, timestamp DESC 
            LIMIT 3
            ''', (f'%{pattern}%', language))
            
            results = cursor.fetchall()
            for response, confidence in results:
                if confidence > 0.7:  # Only high confidence responses
                    suggestions.append(response)
        
        conn.close()
        return suggestions

class WebLearner:
    """Learns from web sources to expand knowledge"""
    
    def __init__(self, db_path="conversation_learning.db"):
        self.db_path = db_path
        self.news_sources = {
            'english': [
                'https://rss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml',
                'https://feeds.npr.org/1001/rss.xml'
            ],
            'kiswahili': [
                'https://www.bbc.com/swahili/index.xml',
                'https://www.dw.com/swahili/rss'
            ]
        }
        
    def fetch_recent_knowledge(self, language: str = 'english', max_articles: int = 10):
        """Fetch recent knowledge from web sources"""
        sources = self.news_sources.get(language, self.news_sources['english'])
        
        knowledge_items = []
        for source_url in sources:
            try:
                feed = feedparser.parse(source_url)
                for entry in feed.entries[:max_articles]:
                    knowledge_item = {
                        'topic': entry.title,
                        'content': self.clean_text(entry.summary if hasattr(entry, 'summary') else entry.title),
                        'source': source_url,
                        'language': language,
                        'relevance_score': self.calculate_relevance(entry.title),
                        'last_updated': datetime.datetime.now(),
                        'hash_key': hashlib.md5(entry.title.encode()).hexdigest()
                    }
                    knowledge_items.append(knowledge_item)
                    
            except Exception as e:
                logger.error(f"Error fetching from {source_url}: {e}")
                
        self.store_knowledge(knowledge_items)
        return len(knowledge_items)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Limit length
        return text[:500] if len(text) > 500 else text
    
    def calculate_relevance(self, title: str) -> float:
        """Calculate relevance score for content"""
        # Simple relevance based on keywords
        relevant_keywords = [
            'technology', 'AI', 'education', 'language', 'communication',
            'teknolojia', 'elimu', 'lugha', 'mawasiliano',
            'ũhinga', 'gũthoma', 'rũgano', 'kwaria',
            'tekno', 'puonjo', 'dhok', 'wuoyo'
        ]
        
        title_lower = title.lower()
        score = sum(1 for keyword in relevant_keywords if keyword in title_lower)
        return min(score / len(relevant_keywords), 1.0)
    
    def store_knowledge(self, knowledge_items: List[Dict]):
        """Store knowledge items in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in knowledge_items:
            cursor.execute('''
            INSERT OR REPLACE INTO knowledge_base 
            (topic, content, source, language, relevance_score, last_updated, hash_key)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (item['topic'], item['content'], item['source'], item['language'],
                  item['relevance_score'], item['last_updated'], item['hash_key']))
        
        conn.commit()
        conn.close()
    
    def search_knowledge(self, query: str, language: str = 'english', limit: int = 5) -> List[Dict]:
        """Search stored knowledge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT topic, content, source, relevance_score FROM knowledge_base 
        WHERE (topic LIKE ? OR content LIKE ?) AND language = ?
        ORDER BY relevance_score DESC, last_updated DESC 
        LIMIT ?
        ''', (f'%{query}%', f'%{query}%', language, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'topic': row[0],
                'content': row[1],
                'source': row[2],
                'relevance_score': row[3]
            }
            for row in results
        ]

class AdaptiveLearningSystem:
    """Main learning system that combines conversation and web learning"""
    
    def __init__(self):
        self.conversation_learner = ConversationLearner()
        self.web_learner = WebLearner()
        self.last_web_update = {}
        
    def learn_from_conversation(self, conversation_id: str, user_message: str,
                              ai_response: str, language: str, confidence: float):
        """Learn from a user conversation"""
        self.conversation_learner.record_conversation(
            conversation_id, user_message, ai_response, language, confidence
        )
        
    def get_enhanced_response(self, user_message: str, language: str, 
                            base_response: str) -> Dict:
        """Get enhanced response using learned knowledge"""
        # Get conversation-based suggestions
        suggestions = self.conversation_learner.get_response_suggestions(user_message, language)
        
        # Search web knowledge
        web_knowledge = self.web_learner.search_knowledge(user_message, language)
        
        # Enhance base response
        enhanced_response = base_response
        additional_info = []
        
        # Add relevant web knowledge
        if web_knowledge:
            relevant_items = [item for item in web_knowledge if item['relevance_score'] > 0.3]
            if relevant_items:
                additional_info.append("Here's some additional relevant information:")
                for item in relevant_items[:2]:  # Limit to 2 items
                    additional_info.append(f"• {item['topic']}: {item['content'][:100]}...")
        
        if additional_info:
            if language == 'kiswahili':
                enhanced_response += "\n\nMaelezo ya ziada:\n" + "\n".join(additional_info)
            elif language == 'kikuyu':
                enhanced_response += "\n\nŨhoro ũngĩ:\n" + "\n".join(additional_info)
            elif language == 'luo':
                enhanced_response += "\n\nWuonruok moko:\n" + "\n".join(additional_info)
            else:
                enhanced_response += "\n\n" + "\n".join(additional_info)
        
        return {
            'enhanced_response': enhanced_response,
            'suggestions_used': len(suggestions),
            'web_knowledge_used': len(web_knowledge),
            'confidence_boost': 0.1 if (suggestions or web_knowledge) else 0.0
        }
    
    def update_web_knowledge(self, language: str = 'english'):
        """Update web knowledge if needed"""
        now = datetime.datetime.now()
        last_update = self.last_web_update.get(language, datetime.datetime.min)
        
        # Update every 6 hours
        if (now - last_update).total_seconds() > 6 * 3600:
            try:
                count = self.web_learner.fetch_recent_knowledge(language)
                self.last_web_update[language] = now
                logger.info(f"Updated {count} knowledge items for {language}")
                return count
            except Exception as e:
                logger.error(f"Failed to update web knowledge: {e}")
                return 0
        return 0
    
    def get_learning_stats(self) -> Dict:
        """Get learning system statistics"""
        conn = sqlite3.connect(self.conversation_learner.db_path)
        cursor = conn.cursor()
        
        # Conversation stats
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT pattern) FROM learned_patterns')
        unique_patterns = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM knowledge_base')
        knowledge_items = cursor.fetchone()[0]
        
        cursor.execute('SELECT language, COUNT(*) FROM conversations GROUP BY language')
        language_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_conversations': total_conversations,
            'unique_patterns': unique_patterns,
            'knowledge_items': knowledge_items,
            'language_distribution': language_distribution,
            'last_web_updates': self.last_web_update
        }

# Global learning system instance
learning_system = AdaptiveLearningSystem()

def enhance_response_with_learning(user_message: str, base_response: str, 
                                 language: str, conversation_id: str, 
                                 confidence: float) -> Dict:
    """Main function to enhance responses with learning"""
    
    # Update web knowledge periodically
    learning_system.update_web_knowledge(language)
    
    # Get enhanced response
    enhanced = learning_system.get_enhanced_response(user_message, language, base_response)
    
    # Record this conversation for future learning
    learning_system.learn_from_conversation(
        conversation_id, user_message, enhanced['enhanced_response'], language, confidence
    )
    
    return {
        'response': enhanced['enhanced_response'],
        'original_response': base_response,
        'learning_applied': enhanced['suggestions_used'] > 0 or enhanced['web_knowledge_used'] > 0,
        'confidence_boost': enhanced['confidence_boost'],
        'suggestions_used': enhanced['suggestions_used'],
        'web_knowledge_used': enhanced['web_knowledge_used']
    }

if __name__ == "__main__":
    # Test the learning system
    system = AdaptiveLearningSystem()
    
    # Test conversation learning
    system.learn_from_conversation(
        "test_conv_1", 
        "What is AI?", 
        "AI stands for Artificial Intelligence...", 
        "english", 
        0.9
    )
    
    # Test web learning
    count = system.update_web_knowledge('english')
    print(f"Fetched {count} knowledge items")
    
    # Test enhanced response
    result = enhance_response_with_learning(
        "Tell me about technology",
        "Technology is advancing rapidly...",
        "english",
        "test_conv_2",
        0.8
    )
    
    print("Enhanced Response:", result['response'])
    print("Learning Applied:", result['learning_applied'])
    
    # Show stats
    stats = system.get_learning_stats()
    print("Learning Stats:", stats)
