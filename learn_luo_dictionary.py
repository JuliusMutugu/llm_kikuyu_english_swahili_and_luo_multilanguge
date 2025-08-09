#!/usr/bin/env python3
"""
Quick Luo Dictionary Learning from Glosbe
Based on the suggestion to use https://glosbe.com/en/luo
"""

import asyncio
import json
import datetime
from online_dictionary_learner import OnlineDictionaryLearner

async def learn_luo_dictionary():
    """
    Specifically learn Luo vocabulary from Glosbe.com
    """
    print("🔍 Learning Luo Dictionary from Glosbe.com")
    print("🌐 Source: https://glosbe.com/en/luo")
    print()
    
    learner = OnlineDictionaryLearner()
    await learner.initialize_session()
    
    # Essential Luo words to start learning
    essential_luo_words = [
        # Greetings and basic phrases
        'inadi', 'ber', 'erokamano', 'mos', 'nying',
        # Family terms
        'mama', 'baba', 'nyako', 'wuod', 'dala', 'ot',
        # Daily life
        'chiemo', 'pi', 'wang', 'odiechieng', 'otieno',
        # Actions
        'dhi', 'bi', 'nindo', 'tich', 'wuok', 'donjo',
        # Numbers and time
        'achiel', 'ariyo', 'adek', 'ang\'wen', 'abich',
        # Nature and environment
        'nam', 'otiep', 'piny', 'polo', 'yamo', 'koth',
        # Emotions and qualities
        'mor', 'kuyo', 'hera', 'ber', 'rach', 'matek'
    ]
    
    print(f"📚 Learning {len(essential_luo_words)} essential Luo words...")
    
    all_entries = []
    
    try:
        # Learn each word from Glosbe
        for i, word in enumerate(essential_luo_words, 1):
            print(f"  {i:2d}. Learning '{word}'...")
            
            try:
                # Get Luo to English translations
                entries = await learner._scrape_glosbe_word(word, 'luo', 'en')
                all_entries.extend(entries)
                
                # Get English to Luo for common translations
                if entries:
                    for entry in entries[:2]:  # Top 2 translations
                        if entry.target_word:
                            reverse_entries = await learner._scrape_glosbe_word(entry.target_word, 'en', 'luo')
                            all_entries.extend(reverse_entries)
                
                # Be respectful to the server
                await asyncio.sleep(1.5)
                
            except Exception as e:
                print(f"    ❌ Error learning '{word}': {e}")
                continue
        
        # Create federated learning data
        if all_entries:
            federated_data = learner.create_federated_learning_data(all_entries)
            
            # Save the data
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"luo_glosbe_dictionary_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(federated_data, f, indent=2, ensure_ascii=False)
            
            print()
            print("✅ Luo Dictionary Learning Complete!")
            print(f"📖 Total entries learned: {len(all_entries)}")
            print(f"🔄 Federated updates created: {len(federated_data['federated_updates'])}")
            print(f"📁 Saved to: {filename}")
            
            # Show sample vocabulary
            print()
            print("📚 Sample Luo Vocabulary Learned:")
            unique_words = {}
            for entry in all_entries:
                if entry.source_word not in unique_words and entry.target_word:
                    unique_words[entry.source_word] = entry.target_word
                    if len(unique_words) >= 10:  # Show top 10
                        break
            
            for luo_word, english_word in unique_words.items():
                print(f"  • {luo_word:12} → {english_word}")
            
            # Quality report
            quality_scores = [entry.quality_score for entry in all_entries]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                print()
                print(f"⭐ Average quality score: {avg_quality:.2f}")
                print(f"🎯 Best quality score: {max(quality_scores):.2f}")
            
            # Integration instructions
            print()
            print("🔗 To integrate with your AI:")
            print("1. Go to Learning Tab → Federated Learning")
            print("2. Add file source:")
            print(f"   URL: {filename}")
            print("   Type: file")
            print("   Trust Level: 0.9")
            print("3. Click 'Sync Now' to apply learning")
            
        else:
            print("❌ No vocabulary entries were successfully learned")
    
    except Exception as e:
        print(f"❌ Error during learning: {e}")
    
    finally:
        await learner.close_session()
    
    return all_entries

async def quick_luo_test():
    """Quick test of a few Luo words"""
    print("🧪 Quick Luo Dictionary Test")
    print()
    
    learner = OnlineDictionaryLearner()
    await learner.initialize_session()
    
    test_words = ['inadi', 'ber', 'erokamano']  # hello, good, thank you
    
    for word in test_words:
        print(f"Testing '{word}'...")
        entries = await learner._scrape_glosbe_word(word, 'luo', 'en')
        
        if entries:
            print(f"  ✅ Found {len(entries)} translations")
            for entry in entries[:3]:
                print(f"    • {entry.source_word} → {entry.target_word}")
        else:
            print(f"  ❌ No translations found")
        
        print()
    
    await learner.close_session()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Quick test mode
        asyncio.run(quick_luo_test())
    else:
        # Full learning mode
        asyncio.run(learn_luo_dictionary())
