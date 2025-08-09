#!/usr/bin/env python3
"""
Learning Analytics Module for Trilingual AI Assistant
Analyzes feedback data and provides insights for continuous improvement
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class LearningAnalytics:
    """
    Analytics engine for processing user feedback and improving AI responses
    """
    
    def __init__(self, data_file: str = None):
        self.data_file = data_file
        self.feedback_data = []
        self.conversation_data = []
        self.language_performance = {}
        self.insights = {}
        
    def load_data(self, json_file: str):
        """Load learning data from exported JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.feedback_data = data.get('feedback_data', [])
            self.conversation_data = data.get('conversations', {})
            self.raw_analytics = data.get('analytics', {})
            
            print(f"✅ Loaded {len(self.feedback_data)} feedback items")
            print(f"✅ Loaded {len(self.conversation_data)} conversations")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_feedback_trends(self):
        """Analyze feedback trends over time"""
        if not self.feedback_data:
            return {}
        
        df = pd.DataFrame(self.feedback_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Daily rating trends
        daily_ratings = df.groupby('date')['rating'].agg(['mean', 'count', 'std']).fillna(0)
        
        # Rating distribution
        rating_dist = df['rating'].value_counts().sort_index()
        
        # Feedback text analysis
        feedback_with_text = df[df['feedback_text'].str.len() > 0]
        
        trends = {
            'daily_ratings': daily_ratings.to_dict(),
            'rating_distribution': rating_dist.to_dict(),
            'average_rating': df['rating'].mean(),
            'total_feedback': len(df),
            'feedback_with_text_ratio': len(feedback_with_text) / len(df) if len(df) > 0 else 0,
            'improvement_trend': self._calculate_improvement_trend(daily_ratings)
        }
        
        return trends
    
    def analyze_language_performance(self):
        """Analyze performance across different languages"""
        if not self.feedback_data:
            return {}
        
        # Extract language from conversation context
        performance_by_lang = {}
        
        for feedback in self.feedback_data:
            chat_id = feedback['chat_id']
            message_idx = feedback['message_index']
            
            # Find the corresponding message in conversations
            if chat_id in self.conversation_data:
                messages = self.conversation_data[chat_id].get('messages', [])
                if message_idx < len(messages):
                    message = messages[message_idx]
                    language = message.get('language', 'unknown')
                    
                    if language not in performance_by_lang:
                        performance_by_lang[language] = []
                    
                    performance_by_lang[language].append(feedback['rating'])
        
        # Calculate statistics for each language
        lang_stats = {}
        for lang, ratings in performance_by_lang.items():
            lang_stats[lang] = {
                'average_rating': np.mean(ratings),
                'total_feedback': len(ratings),
                'rating_std': np.std(ratings),
                'excellent_ratio': len([r for r in ratings if r >= 4]) / len(ratings),
                'poor_ratio': len([r for r in ratings if r <= 2]) / len(ratings)
            }
        
        return lang_stats
    
    def identify_problem_areas(self):
        """Identify areas needing improvement"""
        problems = []
        
        # Low-rated responses
        low_rated = [f for f in self.feedback_data if f['rating'] <= 2]
        if low_rated:
            problems.append({
                'type': 'Low Ratings',
                'count': len(low_rated),
                'description': f"{len(low_rated)} responses rated 2 or below",
                'severity': 'high' if len(low_rated) > len(self.feedback_data) * 0.2 else 'medium'
            })
        
        # Common error types
        if hasattr(self, 'raw_analytics') and 'error_reports' in self.raw_analytics:
            error_types = {}
            for error in self.raw_analytics['error_reports']:
                error_type = error['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                if count >= 3:  # 3 or more of the same error type
                    problems.append({
                        'type': 'Recurring Error',
                        'count': count,
                        'description': f"{count} reports of {error_type}",
                        'severity': 'high' if count > 5 else 'medium'
                    })
        
        # Language-specific issues
        lang_performance = self.analyze_language_performance()
        for lang, stats in lang_performance.items():
            if stats['average_rating'] < 3.0 and stats['total_feedback'] >= 3:
                problems.append({
                    'type': 'Language Performance',
                    'count': stats['total_feedback'],
                    'description': f"{lang} language responses averaging {stats['average_rating']:.1f}/5",
                    'severity': 'high'
                })
        
        return sorted(problems, key=lambda x: x['count'], reverse=True)
    
    def generate_improvement_recommendations(self):
        """Generate specific recommendations for improvement"""
        recommendations = []
        
        # Analyze feedback trends
        trends = self.analyze_feedback_trends()
        lang_performance = self.analyze_language_performance()
        problems = self.identify_problem_areas()
        
        # Rating-based recommendations
        avg_rating = trends.get('average_rating', 0)
        if avg_rating < 3.5:
            recommendations.append({
                'priority': 'high',
                'category': 'Overall Performance',
                'recommendation': 'Focus on improving response quality - average rating is below 3.5',
                'action_items': [
                    'Review low-rated responses for common patterns',
                    'Improve training data quality',
                    'Enhance context understanding'
                ]
            })
        
        # Language-specific recommendations
        for lang, stats in lang_performance.items():
            if stats['poor_ratio'] > 0.3:  # More than 30% poor ratings
                recommendations.append({
                    'priority': 'high',
                    'category': 'Language Performance',
                    'recommendation': f'Improve {lang} language responses - {stats["poor_ratio"]:.0%} rated poorly',
                    'action_items': [
                        f'Collect more training data for {lang}',
                        f'Review {lang} cultural context handling',
                        f'Improve {lang} grammar and syntax'
                    ]
                })
        
        # Error-based recommendations
        high_priority_problems = [p for p in problems if p['severity'] == 'high']
        for problem in high_priority_problems[:3]:  # Top 3 high-priority issues
            if problem['type'] == 'Recurring Error':
                recommendations.append({
                    'priority': 'high',
                    'category': 'Error Reduction',
                    'recommendation': f'Address recurring {problem["description"]}',
                    'action_items': [
                        'Analyze error context patterns',
                        'Implement specific fixes',
                        'Add error prevention logic'
                    ]
                })
        
        # Positive recommendations
        if avg_rating > 4.0:
            recommendations.append({
                'priority': 'medium',
                'category': 'Expansion',
                'recommendation': 'High user satisfaction - consider expanding capabilities',
                'action_items': [
                    'Add new language features',
                    'Implement advanced conversation modes',
                    'Expand to new domains'
                ]
            })
        
        return recommendations
    
    def create_learning_report(self, output_file: str = None):
        """Generate comprehensive learning report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_feedback': len(self.feedback_data),
                'total_conversations': len(self.conversation_data),
                'analysis_period': self._get_analysis_period()
            },
            'feedback_trends': self.analyze_feedback_trends(),
            'language_performance': self.analyze_language_performance(),
            'problem_areas': self.identify_problem_areas(),
            'recommendations': self.generate_improvement_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            print(f"📊 Learning report saved to {output_file}")
        
        return report
    
    def visualize_insights(self, save_plots: bool = False):
        """Create visualizations for learning insights"""
        if not self.feedback_data:
            print("❌ No feedback data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trilingual AI Assistant - Learning Analytics', fontsize=16, fontweight='bold')
        
        # 1. Rating distribution
        df = pd.DataFrame(self.feedback_data)
        rating_counts = df['rating'].value_counts().sort_index()
        
        axes[0, 0].bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating (1-5)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Ratings over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        daily_ratings = df.groupby('date')['rating'].mean()
        
        axes[0, 1].plot(daily_ratings.index, daily_ratings.values, marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Average Rating Trend')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Average Rating')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Language performance
        lang_performance = self.analyze_language_performance()
        if lang_performance:
            languages = list(lang_performance.keys())
            avg_ratings = [lang_performance[lang]['average_rating'] for lang in languages]
            
            bars = axes[1, 0].bar(languages, avg_ratings, color='lightcoral', edgecolor='darkred')
            axes[1, 0].set_title('Average Rating by Language')
            axes[1, 0].set_xlabel('Language')
            axes[1, 0].set_ylabel('Average Rating')
            axes[1, 0].set_ylim(0, 5)
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, rating in zip(bars, avg_ratings):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                              f'{rating:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Feedback frequency over time
        feedback_counts = df.groupby('date').size()
        axes[1, 1].bar(feedback_counts.index, feedback_counts.values, color='lightgreen', edgecolor='darkgreen')
        axes[1, 1].set_title('Daily Feedback Volume')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Number of Feedback')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'learning_analytics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                       dpi=300, bbox_inches='tight')
            print("📊 Visualization saved to PNG file")
        
        plt.show()
    
    def _calculate_improvement_trend(self, daily_ratings):
        """Calculate if ratings are improving over time"""
        if len(daily_ratings) < 2:
            return 'insufficient_data'
        
        recent_avg = daily_ratings['mean'].tail(7).mean()  # Last 7 days
        older_avg = daily_ratings['mean'].head(7).mean()   # First 7 days
        
        if recent_avg > older_avg * 1.1:  # 10% improvement
            return 'improving'
        elif recent_avg < older_avg * 0.9:  # 10% decline
            return 'declining'
        else:
            return 'stable'
    
    def _get_analysis_period(self):
        """Get the analysis time period"""
        if not self.feedback_data:
            return 'No data'
        
        timestamps = [pd.to_datetime(f['timestamp']) for f in self.feedback_data]
        start_date = min(timestamps).date()
        end_date = max(timestamps).date()
        
        return f"{start_date} to {end_date}"

# Example usage and utility functions
def analyze_exported_data(json_file: str):
    """Quick analysis of exported learning data"""
    analytics = LearningAnalytics()
    
    if not analytics.load_data(json_file):
        return None
    
    print("\n🎓 TRILINGUAL AI LEARNING ANALYSIS")
    print("=" * 50)
    
    # Generate report
    report = analytics.create_learning_report()
    
    # Display key insights
    print(f"\n📊 SUMMARY:")
    print(f"• Total Feedback: {report['summary']['total_feedback']}")
    print(f"• Total Conversations: {report['summary']['total_conversations']}")
    print(f"• Analysis Period: {report['summary']['analysis_period']}")
    
    # Feedback trends
    trends = report['feedback_trends']
    print(f"\n📈 FEEDBACK TRENDS:")
    print(f"• Average Rating: {trends['average_rating']:.2f}/5")
    print(f"• Rating Trend: {trends['improvement_trend']}")
    print(f"• Detailed Feedback: {trends['feedback_with_text_ratio']:.0%}")
    
    # Language performance
    lang_perf = report['language_performance']
    if lang_perf:
        print(f"\n🌍 LANGUAGE PERFORMANCE:")
        for lang, stats in lang_perf.items():
            print(f"• {lang}: {stats['average_rating']:.2f}/5 ({stats['total_feedback']} feedback)")
    
    # Problem areas
    problems = report['problem_areas']
    if problems:
        print(f"\n🚨 TOP ISSUES:")
        for problem in problems[:3]:
            print(f"• {problem['type']}: {problem['description']}")
    
    # Recommendations
    recommendations = report['recommendations']
    if recommendations:
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for rec in recommendations[:3]:
            print(f"• {rec['recommendation']}")
    
    # Create visualizations
    analytics.visualize_insights(save_plots=True)
    
    return analytics

if __name__ == "__main__":
    # Example usage
    print("🎓 Trilingual AI Learning Analytics Tool")
    print("=====================================")
    print()
    print("Usage:")
    print("1. Export learning data from the Streamlit app")
    print("2. Run: python learning_analytics.py")
    print("3. Or import and use: analyze_exported_data('your_data.json')")
    print()
    
    # Check for data files in current directory
    data_files = list(Path('.').glob('learning_data_*.json'))
    if data_files:
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 Found data file: {latest_file}")
        print("Analyzing...")
        analyze_exported_data(str(latest_file))
    else:
        print("📁 No learning data files found. Export data from the app first.")
