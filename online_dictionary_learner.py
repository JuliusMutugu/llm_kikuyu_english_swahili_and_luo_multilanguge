#!/usr/bin/env python3
"""
Online Dictionary Learning Module for Trilingual AI Assistant
Automatically learns from online dictionaries like Glosbe for improved language support
"""

import requests
import json
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import time
import datetime
from typing import Dict, List, Any, Optional
import re
import logging
from dataclasses import dataclass
from urllib.parse import urljoin, quote
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DictionaryEntry:
    """Represents a dictionary entry with translations and examples"""
    source_word: str
    target_word: str
    source_language: str
    target_language: str
    definition: str
    examples: List[str]
    frequency: int
    quality_score: float
    cultural_context: str
    source_url: str
    
class OnlineDictionaryLearner:
    """
    Learns vocabulary and expressions from online dictionaries
    """
    
    def __init__(self):
        self.session = None
        self.learned_entries = []
        self.language_mappings = {
            'en': 'english',
            'sw': 'swahili', 
            'ki': 'kikuyu',
            'luo': 'luo'
        }
        self.delay_between_requests = 1  # Be respectful to servers
        
    async def initialize_session(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def scrape_glosbe_dictionary(self, source_lang: str, target_lang: str, max_pages: int = 5) -> List[DictionaryEntry]:
        """
        Scrape dictionary entries from Glosbe
        """
        entries = []
        
        try:
            # Common words to start with for each language
            starter_words = {
                'luo': ['inadi', 'amor', 'nyako', 'wuod', 'dala', 'tiend', 'chiemo', 'pi', 'ber', 'marach'],
                'sw': ['habari', 'rafiki', 'chakula', 'maji', 'nyumba', 'mzuri', 'mbaya', 'asante', 'karibu', 'pole'],
                'ki': ['wi', 'mwega', 'irio', 'mai', 'nyumba', 'wega', 'uru', 'ngai', 'mutumia', 'mundu'],
                'en': ['hello', 'friend', 'food', 'water', 'house', 'good', 'bad', 'thank', 'welcome', 'sorry']
            }
            
            words_to_search = starter_words.get(source_lang, ['hello', 'good', 'thank'])
            
            for word in words_to_search[:10]:  # Limit to first 10 words
                try:
                    word_entries = await self._scrape_glosbe_word(word, source_lang, target_lang)
                    entries.extend(word_entries)
                    
                    # Be respectful - delay between requests
                    await asyncio.sleep(self.delay_between_requests)
                    
                except Exception as e:
                    logger.error(f"Error scraping word '{word}': {e}")
                    continue
            
            logger.info(f"Scraped {len(entries)} entries from Glosbe ({source_lang}->{target_lang})")
            
        except Exception as e:
            logger.error(f"Error scraping Glosbe: {e}")
        
        return entries
    
    async def _scrape_glosbe_word(self, word: str, source_lang: str, target_lang: str) -> List[DictionaryEntry]:
        """Scrape a specific word from Glosbe"""
        entries = []
        
        try:
            # Construct Glosbe URL
            url = f"https://glosbe.com/{source_lang}/{target_lang}/{quote(word)}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract translations
                    translations = self._extract_glosbe_translations(soup, word, source_lang, target_lang, url)
                    entries.extend(translations)
                
        except Exception as e:
            logger.error(f"Error scraping word '{word}' from Glosbe: {e}")
        
        return entries
    
    def _extract_glosbe_translations(self, soup: BeautifulSoup, source_word: str, source_lang: str, target_lang: str, url: str) -> List[DictionaryEntry]:
        """Extract translation data from Glosbe page"""
        entries = []
        
        try:
            # Look for translation containers (Glosbe structure may vary)
            translation_elements = soup.find_all(['div', 'span'], class_=re.compile(r'translation|meaning|phrase'))
            
            # Also try to find by common patterns
            if not translation_elements:
                translation_elements = soup.find_all(text=re.compile(r'[a-zA-Z\s]{3,}'))
            
            # Extract examples/sentences
            example_elements = soup.find_all(['div', 'span'], class_=re.compile(r'example|sentence|usage'))
            examples = [elem.get_text().strip() for elem in example_elements[:5] if elem.get_text().strip()]
            
            # Look for definition or meaning
            definition_elements = soup.find_all(['div', 'span'], class_=re.compile(r'definition|meaning|desc'))
            definition = ""
            if definition_elements:
                definition = definition_elements[0].get_text().strip()
            
            # Create entries from found translations
            for i, elem in enumerate(translation_elements[:10]):  # Limit to first 10
                text = elem.get_text().strip() if hasattr(elem, 'get_text') else str(elem).strip()
                
                # Filter valid translations
                if (len(text) > 2 and len(text) < 100 and 
                    not text.startswith('http') and 
                    not re.match(r'^\d+$', text) and
                    text.lower() != source_word.lower()):
                    
                    entry = DictionaryEntry(
                        source_word=source_word,
                        target_word=text,
                        source_language=source_lang,
                        target_language=target_lang,
                        definition=definition,
                        examples=examples,
                        frequency=10 - i,  # Higher frequency for earlier results
                        quality_score=0.7 + (0.3 * (10 - i) / 10),  # Higher quality for earlier results
                        cultural_context=f"{source_lang}_to_{target_lang}",
                        source_url=url
                    )
                    entries.append(entry)
            
        except Exception as e:
            logger.error(f"Error extracting translations: {e}")
        
        return entries
    
    async def scrape_alternative_sources(self, language: str) -> List[DictionaryEntry]:
        """
        Scrape from alternative dictionary sources
        """
        entries = []
        
        # Alternative sources for different languages
        sources = {
            'luo': [
                'https://www.dholuo.com/',
                'https://en.wiktionary.org/wiki/Category:Luo_language'
            ],
            'sw': [
                'https://en.wiktionary.org/wiki/Category:Swahili_language',
                'https://kamusi.org/'
            ],
            'ki': [
                'https://en.wiktionary.org/wiki/Category:Kikuyu_language'
            ]
        }
        
        for source_url in sources.get(language, []):
            try:
                source_entries = await self._scrape_generic_source(source_url, language)
                entries.extend(source_entries)
                await asyncio.sleep(self.delay_between_requests)
                
            except Exception as e:
                logger.error(f"Error scraping {source_url}: {e}")
        
        return entries
    
    async def _scrape_generic_source(self, url: str, language: str) -> List[DictionaryEntry]:
        """Generic scraper for dictionary websites"""
        entries = []
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for word lists or vocabulary
                    word_elements = soup.find_all(['a', 'span', 'div'], text=re.compile(r'^[a-zA-Z]+$'))
                    
                    for elem in word_elements[:20]:  # Limit extraction
                        word_text = elem.get_text().strip()
                        
                        if len(word_text) > 2 and len(word_text) < 50:
                            entry = DictionaryEntry(
                                source_word=word_text,
                                target_word="",  # Will be filled by translation
                                source_language=language,
                                target_language="en",
                                definition="",
                                examples=[],
                                frequency=5,
                                quality_score=0.6,
                                cultural_context=f"{language}_vocabulary",
                                source_url=url
                            )
                            entries.append(entry)
        
        except Exception as e:
            logger.error(f"Error scraping generic source {url}: {e}")
        
        return entries
    
    def create_federated_learning_data(self, entries: List[DictionaryEntry]) -> Dict[str, Any]:
        """
        Convert dictionary entries to federated learning format
        """
        federated_data = {
            "federated_updates": [],
            "metadata": {
                "source_type": "online_dictionary",
                "extraction_time": datetime.datetime.now().isoformat(),
                "total_entries": len(entries),
                "languages_covered": list(set([e.source_language for e in entries] + [e.target_language for e in entries])),
                "quality_range": {
                    "min": min([e.quality_score for e in entries]) if entries else 0,
                    "max": max([e.quality_score for e in entries]) if entries else 0,
                    "average": sum([e.quality_score for e in entries]) / len(entries) if entries else 0
                }
            }
        }
        
        for entry in entries:
            # Create federated update from dictionary entry
            update = {
                "source_id": f"dictionary_{hashlib.md5(entry.source_url.encode()).hexdigest()[:8]}",
                "language": entry.source_language,
                "update_type": "vocabulary_expansion",
                "data": {
                    "vocabulary_pattern": {
                        "source_word": entry.source_word,
                        "target_translations": [entry.target_word] if entry.target_word else [],
                        "frequency_category": "high" if entry.frequency > 7 else "medium" if entry.frequency > 4 else "low",
                        "definition_available": bool(entry.definition),
                        "examples_count": len(entry.examples)
                    },
                    "language_patterns": {
                        "primary_language": entry.source_language,
                        "translation_language": entry.target_language,
                        "cultural_expressions": [entry.source_word] if entry.source_word else []
                    },
                    "performance_indicators": {
                        "vocabulary_expansion": True,
                        "quality_category": "high" if entry.quality_score > 0.8 else "medium" if entry.quality_score > 0.6 else "low",
                        "cultural_relevance": 0.8  # Dictionary entries are culturally relevant
                    }
                },
                "timestamp": datetime.datetime.now().isoformat(),
                "privacy_hash": hashlib.sha256(f"{entry.source_word}_{entry.target_word}".encode()).hexdigest()[:16],
                "quality_score": entry.quality_score,
                "cultural_context": entry.cultural_context
            }
            
            federated_data["federated_updates"].append(update)
        
        return federated_data
    
    async def learn_from_online_dictionaries(self, languages: List[str] = ['luo', 'sw', 'ki']) -> Dict[str, Any]:
        """
        Main function to learn from online dictionaries
        """
        await self.initialize_session()
        
        all_entries = []
        learning_report = {
            "start_time": datetime.datetime.now().isoformat(),
            "languages_processed": [],
            "total_entries": 0,
            "sources_accessed": [],
            "federated_updates": 0
        }
        
        try:
            for lang in languages:
                logger.info(f"Learning {lang} vocabulary from online sources...")
                
                # Scrape from Glosbe
                glosbe_entries = await self.scrape_glosbe_dictionary(lang, 'en')
                all_entries.extend(glosbe_entries)
                
                # Scrape from alternative sources
                alt_entries = await self.scrape_alternative_sources(lang)
                all_entries.extend(alt_entries)
                
                learning_report["languages_processed"].append(lang)
                
                # Delay between languages
                await asyncio.sleep(2)
            
            # Convert to federated learning format
            federated_data = self.create_federated_learning_data(all_entries)
            
            # Save the learned data
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"online_dictionary_learning_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(federated_data, f, indent=2, ensure_ascii=False)
            
            learning_report.update({
                "end_time": datetime.datetime.now().isoformat(),
                "total_entries": len(all_entries),
                "federated_updates": len(federated_data["federated_updates"]),
                "output_file": filename,
                "sources_accessed": ["glosbe.com", "wiktionary.org"],
                "quality_summary": federated_data["metadata"]["quality_range"]
            })
            
            logger.info(f"Dictionary learning complete! Saved {len(all_entries)} entries to {filename}")
            
        except Exception as e:
            logger.error(f"Error in dictionary learning: {e}")
            learning_report["error"] = str(e)
        
        finally:
            await self.close_session()
        
        return learning_report

# Standalone learning function
async def run_dictionary_learning():
    """Run dictionary learning as standalone process"""
    learner = OnlineDictionaryLearner()
    
    print("🔍 Starting Online Dictionary Learning...")
    print("📚 Sources: Glosbe, Wiktionary, and other dictionary sites")
    print("🌍 Languages: Luo, Kiswahili, Kikuyu")
    print()
    
    report = await learner.learn_from_online_dictionaries()
    
    print("📊 Learning Report:")
    print(f"✅ Languages processed: {', '.join(report['languages_processed'])}")
    print(f"📖 Total entries learned: {report['total_entries']}")
    print(f"🔄 Federated updates created: {report['federated_updates']}")
    print(f"📁 Output file: {report.get('output_file', 'None')}")
    
    if 'quality_summary' in report:
        quality = report['quality_summary']
        print(f"⭐ Quality range: {quality['min']:.2f} - {quality['max']:.2f} (avg: {quality['average']:.2f})")
    
    print("\n🎉 Dictionary learning complete!")
    return report

# Example usage for specific Luo dictionary learning
async def learn_luo_from_glosbe():
    """Specific function to learn Luo from Glosbe"""
    learner = OnlineDictionaryLearner()
    await learner.initialize_session()
    
    print("🔍 Learning Luo vocabulary from Glosbe...")
    
    # Focus on Luo-English dictionary
    luo_entries = await learner.scrape_glosbe_dictionary('luo', 'en', max_pages=10)
    
    # Also get English-Luo for reverse translations
    en_luo_entries = await learner.scrape_glosbe_dictionary('en', 'luo', max_pages=5)
    
    all_entries = luo_entries + en_luo_entries
    
    if all_entries:
        federated_data = learner.create_federated_learning_data(all_entries)
        
        # Save Luo-specific learning data
        filename = f"luo_dictionary_learning_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(federated_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Learned {len(all_entries)} Luo vocabulary entries")
        print(f"📁 Saved to: {filename}")
        
        # Show some examples
        print("\n📖 Sample Luo vocabulary learned:")
        for entry in all_entries[:5]:
            print(f"  • {entry.source_word} → {entry.target_word}")
    
    await learner.close_session()
    return all_entries

if __name__ == "__main__":
    # Run dictionary learning
    asyncio.run(run_dictionary_learning())
