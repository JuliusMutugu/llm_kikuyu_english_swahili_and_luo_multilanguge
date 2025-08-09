#!/usr/bin/env python3
"""
Federated Learning Module for Trilingual AI Assistant
Implements privacy-preserving distributed learning from multiple sources
"""

import json
import requests
import numpy as np
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
import logging
from dataclasses import dataclass, asdict
import threading
import time
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FederatedUpdate:
    """Represents a federated learning update"""
    source_id: str
    language: str
    update_type: str  # 'feedback', 'correction', 'new_data'
    data: Dict[str, Any]
    timestamp: str
    privacy_hash: str
    quality_score: float
    cultural_context: str
    
class FederatedLearningClient:
    """
    Client for federated learning that can connect to multiple sources
    and aggregate learning without sharing raw data
    """
    
    def __init__(self, client_id: str = None):
        self.client_id = client_id or self._generate_client_id()
        self.learning_sources = []
        self.local_updates = []
        self.aggregated_knowledge = {}
        self.privacy_enabled = True
        self.update_interval = 300  # 5 minutes
        self.running = False
        
    def _generate_client_id(self) -> str:
        """Generate unique client ID"""
        timestamp = datetime.datetime.now().isoformat()
        return hashlib.md5(f"trilingual_ai_{timestamp}".encode()).hexdigest()[:12]
    
    def add_learning_source(self, source_config: Dict[str, Any]):
        """Add a federated learning source"""
        source = {
            'id': source_config.get('id', f"source_{len(self.learning_sources)}"),
            'url': source_config.get('url'),
            'type': source_config.get('type', 'api'),  # 'api', 'file', 'database'
            'languages': source_config.get('languages', ['en', 'sw', 'ki', 'luo']),
            'trust_level': source_config.get('trust_level', 0.5),
            'cultural_context': source_config.get('cultural_context', 'general'),
            'update_frequency': source_config.get('update_frequency', 'hourly'),
            'enabled': source_config.get('enabled', True)
        }
        self.learning_sources.append(source)
        logger.info(f"Added learning source: {source['id']}")
    
    def create_privacy_preserving_update(self, data: Dict[str, Any]) -> FederatedUpdate:
        """Create a privacy-preserving update from local data"""
        # Hash sensitive data while preserving learning signal
        privacy_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        
        # Extract learning signals without exposing raw data
        learning_signals = self._extract_learning_signals(data)
        
        update = FederatedUpdate(
            source_id=self.client_id,
            language=data.get('language', 'auto'),
            update_type=data.get('type', 'feedback'),
            data=learning_signals,
            timestamp=datetime.datetime.now().isoformat(),
            privacy_hash=privacy_hash,
            quality_score=self._calculate_quality_score(data),
            cultural_context=data.get('cultural_context', 'general')
        )
        
        return update
    
    def _extract_learning_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learning signals while preserving privacy"""
        signals = {}
        
        # Language patterns (aggregated, not specific)
        if 'feedback' in data:
            signals['feedback_pattern'] = {
                'rating_category': self._categorize_rating(data['feedback'].get('rating', 3)),
                'feedback_length': len(data['feedback'].get('text', '')),
                'error_type': data['feedback'].get('error_type'),
                'improvement_area': data['feedback'].get('improvement_area')
            }
        
        # Language usage patterns
        if 'language_usage' in data:
            signals['language_patterns'] = {
                'primary_language': data['language_usage'].get('primary'),
                'code_switching': data['language_usage'].get('code_switching', False),
                'cultural_expressions': data['language_usage'].get('cultural_expressions', [])
            }
        
        # Performance metrics (aggregated)
        if 'performance' in data:
            signals['performance_indicators'] = {
                'response_time_category': self._categorize_response_time(data['performance'].get('response_time', 0)),
                'accuracy_category': self._categorize_accuracy(data['performance'].get('accuracy', 0)),
                'user_satisfaction': data['performance'].get('satisfaction_score', 0)
            }
        
        return signals
    
    def _categorize_rating(self, rating: float) -> str:
        """Categorize ratings for privacy"""
        if rating >= 4.5:
            return 'excellent'
        elif rating >= 3.5:
            return 'good'
        elif rating >= 2.5:
            return 'average'
        else:
            return 'poor'
    
    def _categorize_response_time(self, time_ms: float) -> str:
        """Categorize response times"""
        if time_ms < 1000:
            return 'fast'
        elif time_ms < 3000:
            return 'medium'
        else:
            return 'slow'
    
    def _categorize_accuracy(self, accuracy: float) -> str:
        """Categorize accuracy scores"""
        if accuracy >= 0.9:
            return 'high'
        elif accuracy >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate quality score for the update"""
        score = 0.5  # Base score
        
        # Boost score based on data quality indicators
        if 'feedback' in data and data['feedback'].get('text'):
            score += 0.2  # Has descriptive feedback
        
        if 'language_usage' in data:
            score += 0.1  # Has language usage data
        
        if 'cultural_context' in data and data['cultural_context'] != 'general':
            score += 0.1  # Has specific cultural context
        
        if 'performance' in data:
            score += 0.1  # Has performance metrics
        
        return min(score, 1.0)
    
    async def fetch_updates_from_source(self, source: Dict[str, Any]) -> List[FederatedUpdate]:
        """Fetch updates from a federated learning source"""
        updates = []
        
        try:
            if source['type'] == 'api':
                updates = await self._fetch_api_updates(source)
            elif source['type'] == 'file':
                updates = await self._fetch_file_updates(source)
            elif source['type'] == 'github':
                updates = await self._fetch_github_updates(source)
            
            logger.info(f"Fetched {len(updates)} updates from {source['id']}")
            
        except Exception as e:
            logger.error(f"Error fetching from {source['id']}: {e}")
        
        return updates
    
    async def _fetch_api_updates(self, source: Dict[str, Any]) -> List[FederatedUpdate]:
        """Fetch updates from API source"""
        updates = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{source['url']}/federated/updates"
                params = {
                    'client_id': self.client_id,
                    'languages': ','.join(source['languages']),
                    'since': self._get_last_update_time(source['id'])
                }
                
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        updates = [FederatedUpdate(**update) for update in data.get('updates', [])]
        
        except Exception as e:
            logger.error(f"API fetch error: {e}")
        
        return updates
    
    async def _fetch_file_updates(self, source: Dict[str, Any]) -> List[FederatedUpdate]:
        """Fetch updates from file source"""
        updates = []
        
        try:
            file_path = Path(source['url'])
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    updates = [FederatedUpdate(**update) for update in data.get('updates', [])]
        
        except Exception as e:
            logger.error(f"File fetch error: {e}")
        
        return updates
    
    async def _fetch_github_updates(self, source: Dict[str, Any]) -> List[FederatedUpdate]:
        """Fetch updates from GitHub repository"""
        updates = []
        
        try:
            # Example: fetch from a GitHub repository with learning data
            repo_url = source['url']
            api_url = f"https://api.github.com/repos/{repo_url}/contents/learning_data"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, timeout=30) as response:
                    if response.status == 200:
                        files = await response.json()
                        
                        for file_info in files:
                            if file_info['name'].endswith('.json'):
                                # Fetch file content
                                content_response = await session.get(file_info['download_url'])
                                if content_response.status == 200:
                                    content = await content_response.json()
                                    if 'federated_updates' in content:
                                        file_updates = [FederatedUpdate(**update) 
                                                      for update in content['federated_updates']]
                                        updates.extend(file_updates)
        
        except Exception as e:
            logger.error(f"GitHub fetch error: {e}")
        
        return updates
    
    def aggregate_updates(self, updates: List[FederatedUpdate]) -> Dict[str, Any]:
        """Aggregate federated updates using privacy-preserving techniques"""
        aggregated = {
            'language_improvements': {},
            'cultural_insights': {},
            'performance_patterns': {},
            'error_corrections': {},
            'update_count': len(updates),
            'aggregation_time': datetime.datetime.now().isoformat()
        }
        
        # Group updates by language
        language_groups = {}
        for update in updates:
            lang = update.language
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(update)
        
        # Aggregate each language group
        for lang, lang_updates in language_groups.items():
            aggregated['language_improvements'][lang] = self._aggregate_language_updates(lang_updates)
        
        # Aggregate cultural insights
        cultural_groups = {}
        for update in updates:
            context = update.cultural_context
            if context not in cultural_groups:
                cultural_groups[context] = []
            cultural_groups[context].append(update)
        
        for context, context_updates in cultural_groups.items():
            aggregated['cultural_insights'][context] = self._aggregate_cultural_updates(context_updates)
        
        return aggregated
    
    def _aggregate_language_updates(self, updates: List[FederatedUpdate]) -> Dict[str, Any]:
        """Aggregate updates for a specific language"""
        if not updates:
            return {}
        
        # Count feedback patterns
        feedback_patterns = {}
        performance_stats = {}
        error_types = {}
        
        for update in updates:
            data = update.data
            
            # Aggregate feedback patterns
            if 'feedback_pattern' in data:
                pattern = data['feedback_pattern']
                rating_cat = pattern.get('rating_category', 'unknown')
                feedback_patterns[rating_cat] = feedback_patterns.get(rating_cat, 0) + 1
                
                error_type = pattern.get('error_type')
                if error_type:
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Aggregate performance indicators
            if 'performance_indicators' in data:
                perf = data['performance_indicators']
                for metric, value in perf.items():
                    if metric not in performance_stats:
                        performance_stats[metric] = []
                    performance_stats[metric].append(value)
        
        return {
            'feedback_distribution': feedback_patterns,
            'common_errors': error_types,
            'performance_trends': performance_stats,
            'update_count': len(updates),
            'quality_score': np.mean([u.quality_score for u in updates])
        }
    
    def _aggregate_cultural_updates(self, updates: List[FederatedUpdate]) -> Dict[str, Any]:
        """Aggregate cultural context updates"""
        if not updates:
            return {}
        
        cultural_patterns = {}
        language_usage = {}
        
        for update in updates:
            data = update.data
            
            if 'language_patterns' in data:
                patterns = data['language_patterns']
                
                # Track code-switching patterns
                if patterns.get('code_switching'):
                    cultural_patterns['code_switching'] = cultural_patterns.get('code_switching', 0) + 1
                
                # Track cultural expressions
                expressions = patterns.get('cultural_expressions', [])
                for expr in expressions:
                    cultural_patterns[f'expression_{expr}'] = cultural_patterns.get(f'expression_{expr}', 0) + 1
        
        return {
            'cultural_patterns': cultural_patterns,
            'language_usage': language_usage,
            'update_count': len(updates)
        }
    
    def apply_federated_learning(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply federated learning insights to improve the model"""
        improvements = {
            'language_adjustments': {},
            'cultural_enhancements': {},
            'error_corrections': {},
            'performance_optimizations': {},
            'applied_at': datetime.datetime.now().isoformat()
        }
        
        # Apply language improvements
        for lang, lang_data in aggregated_data.get('language_improvements', {}).items():
            improvements['language_adjustments'][lang] = self._apply_language_improvements(lang, lang_data)
        
        # Apply cultural enhancements
        for context, context_data in aggregated_data.get('cultural_insights', {}).items():
            improvements['cultural_enhancements'][context] = self._apply_cultural_improvements(context, context_data)
        
        return improvements
    
    def _apply_language_improvements(self, language: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply language-specific improvements"""
        adjustments = {
            'priority_areas': [],
            'confidence_adjustments': {},
            'error_prevention': []
        }
        
        # Identify priority improvement areas
        feedback_dist = data.get('feedback_distribution', {})
        if feedback_dist.get('poor', 0) > feedback_dist.get('excellent', 0):
            adjustments['priority_areas'].append('response_quality')
        
        # Common errors to address
        common_errors = data.get('common_errors', {})
        for error_type, count in common_errors.items():
            if count >= 3:  # Threshold for addressing errors
                adjustments['error_prevention'].append({
                    'error_type': error_type,
                    'frequency': count,
                    'priority': 'high' if count > 5 else 'medium'
                })
        
        # Performance adjustments
        performance = data.get('performance_trends', {})
        if 'accuracy_category' in performance:
            accuracy_levels = performance['accuracy_category']
            low_accuracy_count = accuracy_levels.count('low')
            if low_accuracy_count > len(accuracy_levels) * 0.3:  # More than 30% low accuracy
                adjustments['confidence_adjustments']['lower_confidence'] = True
        
        return adjustments
    
    def _apply_cultural_improvements(self, context: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cultural context improvements"""
        enhancements = {
            'cultural_awareness': [],
            'expression_patterns': {},
            'code_switching_support': False
        }
        
        cultural_patterns = data.get('cultural_patterns', {})
        
        # Code-switching support
        if cultural_patterns.get('code_switching', 0) > 0:
            enhancements['code_switching_support'] = True
        
        # Cultural expressions to learn
        for pattern, count in cultural_patterns.items():
            if pattern.startswith('expression_') and count >= 2:
                expression = pattern.replace('expression_', '')
                enhancements['expression_patterns'][expression] = count
        
        return enhancements
    
    def _get_last_update_time(self, source_id: str) -> str:
        """Get last update time for a source"""
        # In a real implementation, this would be stored persistently
        return (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat()
    
    async def start_federated_learning(self):
        """Start the federated learning process"""
        self.running = True
        logger.info("Starting federated learning client...")
        
        while self.running:
            try:
                # Fetch updates from all sources
                all_updates = []
                
                for source in self.learning_sources:
                    if source['enabled']:
                        updates = await self.fetch_updates_from_source(source)
                        all_updates.extend(updates)
                
                if all_updates:
                    # Aggregate updates
                    aggregated = self.aggregate_updates(all_updates)
                    
                    # Apply learning
                    improvements = self.apply_federated_learning(aggregated)
                    
                    # Store results
                    self.aggregated_knowledge.update(aggregated)
                    
                    logger.info(f"Applied {len(all_updates)} federated updates")
                    
                    # Save results
                    await self.save_federated_results(improvements)
                
                # Wait for next iteration
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Federated learning error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def save_federated_results(self, improvements: Dict[str, Any]):
        """Save federated learning results"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"federated_learning_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(improvements, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Federated learning results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def stop_federated_learning(self):
        """Stop the federated learning process"""
        self.running = False
        logger.info("Stopping federated learning client...")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current federated learning status"""
        return {
            'client_id': self.client_id,
            'running': self.running,
            'sources_count': len(self.learning_sources),
            'active_sources': len([s for s in self.learning_sources if s['enabled']]),
            'local_updates': len(self.local_updates),
            'last_aggregation': self.aggregated_knowledge.get('aggregation_time', 'Never'),
            'update_interval': self.update_interval
        }


class FederatedLearningCoordinator:
    """
    Coordinates federated learning across multiple clients
    """
    
    def __init__(self):
        self.clients = {}
        self.global_model_updates = []
        self.coordination_log = []
    
    def register_client(self, client: FederatedLearningClient):
        """Register a federated learning client"""
        self.clients[client.client_id] = client
        logger.info(f"Registered client: {client.client_id}")
    
    def coordinate_learning_round(self) -> Dict[str, Any]:
        """Coordinate a round of federated learning across all clients"""
        round_results = {
            'round_id': f"round_{len(self.coordination_log) + 1}",
            'timestamp': datetime.datetime.now().isoformat(),
            'participating_clients': len(self.clients),
            'aggregated_improvements': {}
        }
        
        # Collect updates from all clients
        all_client_knowledge = []
        for client_id, client in self.clients.items():
            if client.aggregated_knowledge:
                all_client_knowledge.append(client.aggregated_knowledge)
        
        # Global aggregation
        if all_client_knowledge:
            global_aggregation = self._global_aggregate(all_client_knowledge)
            round_results['aggregated_improvements'] = global_aggregation
        
        self.coordination_log.append(round_results)
        return round_results
    
    def _global_aggregate(self, client_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform global aggregation across all clients"""
        global_agg = {
            'languages': {},
            'cultural_contexts': {},
            'global_patterns': {},
            'consensus_improvements': {}
        }
        
        # Aggregate language improvements across clients
        all_language_data = {}
        for knowledge in client_knowledge:
            lang_improvements = knowledge.get('language_improvements', {})
            for lang, data in lang_improvements.items():
                if lang not in all_language_data:
                    all_language_data[lang] = []
                all_language_data[lang].append(data)
        
        # Find consensus improvements
        for lang, lang_data_list in all_language_data.items():
            consensus = self._find_consensus(lang_data_list)
            global_agg['languages'][lang] = consensus
        
        return global_agg
    
    def _find_consensus(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find consensus patterns across multiple data sources"""
        consensus = {
            'common_patterns': {},
            'confidence_level': 0.0,
            'data_sources': len(data_list)
        }
        
        # Find patterns that appear in multiple sources
        all_patterns = {}
        for data in data_list:
            for pattern_type, patterns in data.items():
                if isinstance(patterns, dict):
                    for pattern, count in patterns.items():
                        key = f"{pattern_type}_{pattern}"
                        if key not in all_patterns:
                            all_patterns[key] = []
                        all_patterns[key].append(count)
        
        # Keep patterns with high consensus
        for pattern, counts in all_patterns.items():
            if len(counts) >= len(data_list) * 0.6:  # 60% consensus threshold
                consensus['common_patterns'][pattern] = {
                    'frequency': sum(counts),
                    'sources': len(counts),
                    'consensus_score': len(counts) / len(data_list)
                }
        
        consensus['confidence_level'] = len(consensus['common_patterns']) / max(len(all_patterns), 1)
        
        return consensus


# Example federated learning sources configuration
EXAMPLE_FEDERATED_SOURCES = [
    {
        'id': 'community_feedback',
        'url': 'https://api.example.com/trilingual-feedback',
        'type': 'api',
        'languages': ['en', 'sw', 'ki', 'luo'],
        'trust_level': 0.8,
        'cultural_context': 'kenyan_community',
        'update_frequency': 'hourly',
        'enabled': True
    },
    {
        'id': 'academic_research',
        'url': 'research-data/multilingual_improvements.json',
        'type': 'file',
        'languages': ['en', 'sw', 'ki', 'luo'],
        'trust_level': 0.9,
        'cultural_context': 'academic',
        'update_frequency': 'daily',
        'enabled': True
    },
    {
        'id': 'github_community',
        'url': 'your-org/trilingual-ai-learning',
        'type': 'github',
        'languages': ['en', 'sw', 'ki', 'luo'],
        'trust_level': 0.7,
        'cultural_context': 'developer_community',
        'update_frequency': 'daily',
        'enabled': True
    }
]

# Usage example
async def main():
    """Example usage of federated learning"""
    
    # Create federated learning client
    client = FederatedLearningClient()
    
    # Add learning sources
    for source_config in EXAMPLE_FEDERATED_SOURCES:
        client.add_learning_source(source_config)
    
    # Create sample local update
    sample_feedback = {
        'type': 'feedback',
        'language': 'sw',
        'feedback': {
            'rating': 4,
            'text': 'Great response but could improve cultural context',
            'error_type': 'cultural_context',
            'improvement_area': 'cultural_expressions'
        },
        'cultural_context': 'kenyan_community'
    }
    
    # Create privacy-preserving update
    update = client.create_privacy_preserving_update(sample_feedback)
    client.local_updates.append(update)
    
    print("Federated Learning Client Status:")
    print(json.dumps(client.get_learning_status(), indent=2))
    
    # Start federated learning (would run continuously in practice)
    # await client.start_federated_learning()

if __name__ == "__main__":
    asyncio.run(main())
