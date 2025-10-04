import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import json
import base64
from io import BytesIO, StringIO
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
from datetime import datetime
import warnings
import zipfile as zf
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs('uploads', exist_ok=True)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Global variables to store current analysis results
current_analysis_results = None
current_charts = None
current_insights = None

class SentimentAnalysisEngine:
    def __init__(self):
        self.positive_words = set([
            'excellent', 'outstanding', 'fantastic', 'amazing', 'brilliant', 'superb',
            'great', 'good', 'helpful', 'valuable', 'useful', 'informative', 'engaging',
            'comprehensive', 'thorough', 'clear', 'practical', 'relevant', 'effective',
            'perfect', 'wonderful', 'inspiring', 'loved', 'incredible', 'awesome'
        ])

        self.negative_words = set([
            'terrible', 'awful', 'poor', 'bad', 'disappointing', 'boring', 'unclear',
            'confusing', 'useless', 'waste', 'frustrating', 'unprepared', 'disorganized',
            'outdated', 'irrelevant', 'superficial', 'rushed', 'monotonous', 'hate',
            'horrible', 'worst', 'failed', 'disaster'
        ])

    def analyze_sentiment_textblob(self, text):
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }

    def analyze_sentiment_vader(self, text):
        """VADER sentiment analysis"""
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'sentiment': sentiment,
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }

    def analyze_sentiment_custom(self, text):
        """Custom rule-based sentiment analysis"""
        words = re.findall(r'\b\w+\b', text.lower())

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        if positive_count > negative_count:
            sentiment = 'Positive'
            confidence = min(0.9, 0.5 + (positive_count - negative_count) / len(words))
        elif negative_count > positive_count:
            sentiment = 'Negative'
            confidence = min(0.9, 0.5 + (negative_count - positive_count) / len(words))
        else:
            sentiment = 'Neutral'
            confidence = 0.5

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_words': positive_count,
            'negative_words': negative_count
        }

    def find_feedback_column(self, df):
        """Automatically find the feedback text column"""
        possible_names = ['feedback', 'review', 'comment', 'text', 'response', 'opinion', 'message']

        for col in df.columns:
            if any(name in col.lower() for name in possible_names):
                return col

        # Fallback: find column with longest average text length
        text_lengths = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                text_lengths[col] = avg_length

        if text_lengths:
            return max(text_lengths, key=text_lengths.get)

        return df.columns[0]  # Last resort

    def analyze_dataset(self, df):
        """Perform comprehensive sentiment analysis on dataset"""
        feedback_col = self.find_feedback_column(df)

        if feedback_col not in df.columns:
            raise ValueError(f"Could not find feedback column: {feedback_col}")

        results = []

        for idx, row in df.iterrows():
            feedback_text = str(row[feedback_col])

            # Apply all three sentiment analysis methods
            textblob_result = self.analyze_sentiment_textblob(feedback_text)
            vader_result = self.analyze_sentiment_vader(feedback_text)
            custom_result = self.analyze_sentiment_custom(feedback_text)

            # Combine results
            result = {
                'original_index': idx,
                'feedback_text': feedback_text,
                'textblob_sentiment': textblob_result['sentiment'],
                'textblob_polarity': textblob_result['polarity'],
                'textblob_subjectivity': textblob_result['subjectivity'],
                'vader_sentiment': vader_result['sentiment'],
                'vader_compound': vader_result['compound'],
                'custom_sentiment': custom_result['sentiment'],
                'custom_confidence': custom_result['confidence']
            }

            # Add other columns from original data
            for col in df.columns:
                if col != feedback_col:
                    result[f'original_{col}'] = row[col]

            results.append(result)

        return pd.DataFrame(results), feedback_col

    def generate_insights(self, df):
        """Generate AI-powered insights from analysis results"""
        total_feedback = len(df)

        # Overall sentiment distribution
        textblob_positive = (df['textblob_sentiment'] == 'Positive').sum()
        textblob_negative = (df['textblob_sentiment'] == 'Negative').sum()
        textblob_neutral = (df['textblob_sentiment'] == 'Neutral').sum()

        positive_pct = round(textblob_positive / total_feedback * 100, 1)
        negative_pct = round(textblob_negative / total_feedback * 100, 1)

        # Advanced insights
        avg_polarity = df['textblob_polarity'].mean()
        avg_subjectivity = df['textblob_subjectivity'].mean()

        insights = []

        # Insight 1: Overall sentiment
        if positive_pct >= 70:
            sentiment_level = "excellent"
            priority = "success"
        elif positive_pct >= 50:
            sentiment_level = "good"
            priority = "info"
        else:
            sentiment_level = "concerning"
            priority = "danger"

        insights.append({
            'icon': '📊',
            'title': 'Overall Sentiment Analysis',
            'text': f'Analysis shows {positive_pct}% positive feedback, indicating {sentiment_level} satisfaction levels across {total_feedback} responses.',
            'priority': priority,
            'metric': f'{positive_pct}%',
            'trend': 'up' if positive_pct >= 60 else 'down',
            'detailed_analysis': f'Out of {total_feedback} total feedback entries, {textblob_positive} were classified as positive ({positive_pct}%), {textblob_negative} as negative ({negative_pct}%), and {textblob_neutral} as neutral. This distribution indicates {sentiment_level} overall satisfaction with the analyzed content.'
        })

        # Insight 2: Sentiment distribution
        dominant_sentiment = max([
            ('Positive', textblob_positive),
            ('Negative', textblob_negative), 
            ('Neutral', textblob_neutral)
        ], key=lambda x: x[1])

        insights.append({
            'icon': '🎯',
            'title': 'Dominant Sentiment Pattern',
            'text': f'{dominant_sentiment[0]} sentiment dominates with {dominant_sentiment[1]} responses ({round(dominant_sentiment[1]/total_feedback*100, 1)}%). This indicates clear sentiment direction.',
            'priority': 'info',
            'metric': f'{dominant_sentiment[1]} responses',
            'trend': 'stable',
            'detailed_analysis': f'The dominant sentiment pattern shows that {dominant_sentiment[0].lower()} responses represent the majority opinion. This clear dominance suggests consistent patterns in the feedback, which can be valuable for understanding overall trends and making informed decisions.'
        })

        # Insight 3: Polarity strength
        if abs(avg_polarity) > 0.3:
            polarity_strength = "strong"
            polarity_priority = "success"
        elif abs(avg_polarity) > 0.1:
            polarity_strength = "moderate"
            polarity_priority = "info"
        else:
            polarity_strength = "weak"
            polarity_priority = "warning"

        insights.append({
            'icon': '⚡',
            'title': 'Sentiment Intensity Analysis',
            'text': f'Average sentiment polarity is {avg_polarity:.3f}, indicating {polarity_strength} emotional responses in the feedback.',
            'priority': polarity_priority,
            'metric': f'{avg_polarity:.3f}',
            'trend': 'up' if avg_polarity > 0 else 'down',
            'detailed_analysis': f'The average polarity score of {avg_polarity:.3f} indicates {polarity_strength} emotional intensity in the feedback. Scores closer to +1 or -1 represent stronger emotional responses, while scores near 0 indicate more neutral or balanced feedback. This level of emotional intensity suggests passionate engagement from respondents.' if abs(avg_polarity) > 0.3 else f'This level of emotional intensity suggests measured responses from respondents.' if abs(avg_polarity) > 0.1 else f'This level of emotional intensity suggests neutral engagement from respondents.'
        })

        # Insight 4: Subjectivity analysis
        if avg_subjectivity > 0.6:
            subjectivity_level = "highly subjective"
            subj_priority = "info"
            subj_description = "opinion-driven with personal perspectives"
        elif avg_subjectivity > 0.4:
            subjectivity_level = "moderately subjective"
            subj_priority = "info"
            subj_description = "balanced between opinions and facts"
        else:
            subjectivity_level = "objective"
            subj_priority = "success"
            subj_description = "fact-based with minimal emotional content"

        insights.append({
            'icon': '🔍',
            'title': 'Content Subjectivity Analysis',
            'text': f'Feedback is {subjectivity_level} (score: {avg_subjectivity:.3f}), showing the emotional vs factual nature of responses.',
            'priority': subj_priority,
            'metric': f'{avg_subjectivity:.3f}',
            'trend': 'stable',
            'detailed_analysis': f'The subjectivity score of {avg_subjectivity:.3f} indicates that the feedback is {subjectivity_level}. Highly subjective feedback (>0.6) contains more personal opinions and emotional expressions, while objective feedback (<0.4) contains more factual statements. This level suggests that responses are {subj_description}.'
        })

        # Insight 5: Improvement recommendations
        if negative_pct > 30:
            recommendation = "Immediate attention needed to address negative feedback patterns"
            rec_priority = "danger"
            action_plan = "Focus on identifying and resolving the root causes of dissatisfaction. Implement immediate corrective measures and follow-up surveys to track improvement."
        elif negative_pct > 15:
            recommendation = "Monitor negative trends and implement targeted improvements"
            rec_priority = "warning"
            action_plan = "Analyze negative feedback for common themes, develop targeted improvement strategies, and implement changes while monitoring progress."
        else:
            recommendation = "Maintain current positive momentum while monitoring feedback quality"
            rec_priority = "success"
            action_plan = "Continue current successful practices, implement minor optimizations based on feedback, and maintain regular monitoring to preserve high satisfaction levels."

        insights.append({
            'icon': '💡',
            'title': 'AI-Powered Recommendations',
            'text': recommendation,
            'priority': rec_priority,
            'metric': f'{negative_pct}% negative',
            'trend': 'down' if negative_pct < 20 else 'up',
            'detailed_analysis': f'Based on the analysis, {action_plan} The current negative feedback rate of {negative_pct}% requires immediate intervention.' if negative_pct > 30 else f'Based on the analysis, {action_plan} The current negative feedback rate of {negative_pct}% suggests room for improvement.' if negative_pct > 15 else f'Based on the analysis, {action_plan} The current negative feedback rate of {negative_pct}% indicates strong performance. Regular monitoring and proactive management will help maintain or improve satisfaction levels.'
        })

        return insights

# Initialize sentiment engine
sentiment_engine = SentimentAnalysisEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_feedback():
    global current_analysis_results, current_charts, current_insights

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and file.filename.lower().endswith('.csv'):
            # Create unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            # Read and analyze data
            df = pd.read_csv(filepath)

            if df.empty:
                return jsonify({'error': 'Empty dataset'}), 400

            # Perform analysis
            analyzed_df, feedback_col = sentiment_engine.analyze_dataset(df)

            # Generate insights
            insights = sentiment_engine.generate_insights(analyzed_df)

            # Create visualizations
            charts = create_visualizations(analyzed_df)

            # Store results globally for downloads
            current_analysis_results = analyzed_df
            current_charts = charts
            current_insights = insights

            # Clean up uploaded file
            os.remove(filepath)

            return jsonify({
                'success': True,
                'insights': insights,
                'charts': charts,
                'stats': {
                    'total_feedback': len(analyzed_df),
                    'positive_percent': round((analyzed_df['textblob_sentiment'] == 'Positive').sum() / len(analyzed_df) * 100, 1),
                    'negative_percent': round((analyzed_df['textblob_sentiment'] == 'Negative').sum() / len(analyzed_df) * 100, 1),
                    'neutral_percent': round((analyzed_df['textblob_sentiment'] == 'Neutral').sum() / len(analyzed_df) * 100, 1),
                    'average_polarity': round(analyzed_df['textblob_polarity'].mean(), 3),
                    'average_subjectivity': round(analyzed_df['textblob_subjectivity'].mean(), 3)
                }
            })

        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/demo')
def demo_analysis():
    global current_analysis_results, current_charts, current_insights

    try:
        # Create demo dataset
        demo_data = {
            'feedback': [
                "The workshop content was extremely comprehensive and well-structured. I learned practical skills that I can apply immediately.",
                "Fantastic workshop! The instructor was knowledgeable and engaging. The hands-on exercises were particularly valuable.",
                "Great learning experience with excellent real-world examples. The pace was perfect and materials were top-quality.",
                "Outstanding workshop content and delivery. The interactive sessions made complex topics easy to understand.",
                "Excellent workshop! Very informative and practical. The instructor answered all questions thoroughly.",
                "The workshop was okay but felt rushed. Some topics could have been explained more clearly.",
                "Content was good but the presentation style was quite boring. More interactive elements would help.",
                "Average workshop. Some sections were useful but others felt repetitive and could be condensed.",
                "The workshop was fine overall but the examples used were somewhat outdated and not very relevant.",
                "Decent content but the delivery was monotonous. The instructor seemed unprepared for questions.",
                "Disappointing workshop. The content was too basic and didn't meet my expectations.",
                "Very poor experience. The workshop was disorganized and the instructor was unclear in explanations.",
                "Terrible workshop! Complete waste of time. The material was outdated and irrelevant.",
                "Extremely dissatisfied. The workshop lacked depth and practical applications. Very disappointing.",
                "Awful experience. The instructor was unprofessional and the content was poorly structured.",
                "The workshop exceeded all my expectations. Brilliant instructor with deep expertise in the subject.",
                "Absolutely loved this workshop! Best learning experience I've had. Highly recommend to everyone.",
                "Incredible workshop with amazing insights. The instructor's teaching style was exceptional.",
                "Perfect balance of theory and practice. The workshop materials were excellent and well-organized.",
                "Superb workshop! Learned so much in such a short time. The instructor was inspiring."
            ],
            'workshop_type': ['Data Science', 'Machine Learning', 'Python Programming', 'Leadership', 'Business Analytics'] * 4,
            'instructor': ['Dr. Sarah Johnson', 'Prof. Michael Chen', 'Dr. Emily Rodriguez', 'Prof. David Wilson'] * 5
        }

        df = pd.DataFrame(demo_data)

        # Perform analysis
        analyzed_df, feedback_col = sentiment_engine.analyze_dataset(df)

        # Generate insights
        insights = sentiment_engine.generate_insights(analyzed_df)

        # Create visualizations
        charts = create_visualizations(analyzed_df)

        # Store results globally for downloads
        current_analysis_results = analyzed_df
        current_charts = charts
        current_insights = insights

        return jsonify({
            'success': True,
            'insights': insights,
            'charts': charts,
            'stats': {
                'total_feedback': len(analyzed_df),
                'positive_percent': round((analyzed_df['textblob_sentiment'] == 'Positive').sum() / len(analyzed_df) * 100, 1),
                'negative_percent': round((analyzed_df['textblob_sentiment'] == 'Negative').sum() / len(analyzed_df) * 100, 1),
                'neutral_percent': round((analyzed_df['textblob_sentiment'] == 'Neutral').sum() / len(analyzed_df) * 100, 1),
                'average_polarity': round(analyzed_df['textblob_polarity'].mean(), 3),
                'average_subjectivity': round(analyzed_df['textblob_subjectivity'].mean(), 3)
            }
        })

    except Exception as e:
        print(f"Demo error: {str(e)}")
        return jsonify({'error': f'Demo failed: {str(e)}'}), 500

def create_visualizations(df):
    """Create all visualization charts"""
    charts = {}

    # Set style for matplotlib
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Sentiment Distribution Pie Chart
    sentiment_counts = df['textblob_sentiment'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#28a745', '#dc3545', '#ffc107']  # Green, Red, Yellow
    wedges, texts, autotexts = ax.pie(sentiment_counts.values, 
                                     labels=sentiment_counts.index,
                                     colors=colors,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     textprops={'fontsize': 12})

    ax.set_title('Sentiment Distribution Analysis', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    charts['sentiment_pie'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # 2. Category Performance (if available)
    category_cols = [col for col in df.columns if 'original_' in col and col not in ['original_feedback']]

    if category_cols:
        category_col = category_cols[0]
        if df[category_col].nunique() <= 10:  # Only if reasonable number of categories
            category_sentiment = pd.crosstab(df[category_col], df['textblob_sentiment'])
            category_pct = category_sentiment.div(category_sentiment.sum(axis=1), axis=0) * 100

            fig, ax = plt.subplots(figsize=(14, 10))
            category_pct['Positive'].plot(kind='barh', ax=ax, color='#28a745')
            ax.set_title('Positive Sentiment Percentage by Category', fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Positive Sentiment (%)', fontsize=14)
            ax.set_ylabel('Category', fontsize=14)
            plt.tight_layout()

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            charts['category_bar'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

    # 3. Polarity Distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(df['textblob_polarity'], bins=25, color='skyblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', alpha=0.8, label='Neutral Line')
    ax.set_title('Sentiment Polarity Distribution', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Polarity Score (-1 = Negative, +1 = Positive)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    charts['polarity_hist'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # 4. Word Frequency Analysis
    all_feedback = ' '.join(df['feedback_text']).lower()
    words = re.findall(r'\b[a-zA-Z]+\b', all_feedback)

    # Filter out common stop words
    stop_words = {'the', 'and', 'was', 'were', 'are', 'is', 'to', 'of', 'a', 'an', 
                 'for', 'with', 'on', 'in', 'at', 'by', 'very', 'but', 'it', 'that', 'this'}
    words = [word for word in words if word not in stop_words and len(word) > 3]

    word_counts = Counter(words)
    top_words = dict(word_counts.most_common(20))

    if top_words:  # Only create chart if we have words
        fig, ax = plt.subplots(figsize=(14, 10))
        bars = ax.barh(list(top_words.keys()), list(top_words.values()), color='lightcoral')
        ax.set_title('Most Frequent Words in Feedback', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Frequency', fontsize=14)
        ax.set_ylabel('Words', fontsize=14)

        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontsize=10)

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        charts['word_freq'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

    # 5. Sentiment Comparison Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # TextBlob vs VADER comparison
    textblob_counts = df['textblob_sentiment'].value_counts()
    vader_counts = df['vader_sentiment'].value_counts()

    methods = ['TextBlob', 'VADER']
    positive_vals = [textblob_counts.get('Positive', 0), vader_counts.get('Positive', 0)]
    negative_vals = [textblob_counts.get('Negative', 0), vader_counts.get('Negative', 0)]
    neutral_vals = [textblob_counts.get('Neutral', 0), vader_counts.get('Neutral', 0)]

    x = range(len(methods))
    width = 0.25

    ax1.bar([i - width for i in x], positive_vals, width, label='Positive', color='#28a745')
    ax1.bar(x, neutral_vals, width, label='Neutral', color='#ffc107')  
    ax1.bar([i + width for i in x], negative_vals, width, label='Negative', color='#dc3545')

    ax1.set_title('Sentiment Analysis Method Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Analysis Method', fontsize=12)
    ax1.set_ylabel('Number of Responses', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()

    # Polarity vs Subjectivity scatter
    ax2.scatter(df['textblob_polarity'], df['textblob_subjectivity'], 
               alpha=0.6, c=df['textblob_polarity'], cmap='RdYlGn')
    ax2.set_title('Polarity vs Subjectivity Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Polarity (Negative ← → Positive)', fontsize=12)
    ax2.set_ylabel('Subjectivity (Objective ← → Subjective)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    charts['comparison'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return charts

@app.route('/download-charts')
def download_charts():
    """Download all charts as a ZIP file"""
    global current_charts

    if not current_charts:
        return jsonify({'error': 'No charts available for download'}), 404

    try:
        # Create a ZIP file in memory
        zip_buffer = BytesIO()

        with zf.ZipFile(zip_buffer, 'w', zf.ZIP_DEFLATED) as zip_file:
            chart_titles = {
                'sentiment_pie': 'Sentiment_Distribution_Analysis.png',
                'category_bar': 'Category_Performance_Analysis.png',
                'polarity_hist': 'Polarity_Distribution_Chart.png',
                'word_freq': 'Word_Frequency_Analysis.png',
                'comparison': 'Sentiment_Method_Comparison.png'
            }

            for key, base64_data in current_charts.items():
                if key in chart_titles:
                    # Decode base64 to binary
                    image_data = base64.b64decode(base64_data)
                    # Add to ZIP
                    zip_file.writestr(chart_titles[key], image_data)

        zip_buffer.seek(0)

        # Return the ZIP file
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'sentiment_analysis_charts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )

    except Exception as e:
        print(f"Chart download error: {str(e)}")
        return jsonify({'error': 'Failed to create chart download'}), 500

@app.route('/download-insights')
def download_insights():
    """Download insights as a formatted text file"""
    global current_insights, current_analysis_results

    if not current_insights:
        return jsonify({'error': 'No insights available for download'}), 404

    try:
        # Create insights text content
        insights_content = []
        insights_content.append("🤖 AI-POWERED SENTIMENT ANALYSIS INSIGHTS")
        insights_content.append("=" * 60)
        insights_content.append(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        insights_content.append("")

        if current_analysis_results is not None:
            total_feedback = len(current_analysis_results)
            positive_count = (current_analysis_results['textblob_sentiment'] == 'Positive').sum()
            negative_count = (current_analysis_results['textblob_sentiment'] == 'Negative').sum()
            neutral_count = (current_analysis_results['textblob_sentiment'] == 'Neutral').sum()
            avg_polarity = current_analysis_results['textblob_polarity'].mean()
            avg_subjectivity = current_analysis_results['textblob_subjectivity'].mean()

            insights_content.append("📊 ANALYSIS SUMMARY")
            insights_content.append("-" * 30)
            insights_content.append(f"Total Feedback Entries: {total_feedback}")
            insights_content.append(f"Positive Responses: {positive_count} ({positive_count/total_feedback*100:.1f}%)")
            insights_content.append(f"Negative Responses: {negative_count} ({negative_count/total_feedback*100:.1f}%)")
            insights_content.append(f"Neutral Responses: {neutral_count} ({neutral_count/total_feedback*100:.1f}%)")
            insights_content.append(f"Average Polarity Score: {avg_polarity:.3f}")
            insights_content.append(f"Average Subjectivity Score: {avg_subjectivity:.3f}")
            insights_content.append("")

        insights_content.append("🎯 KEY INSIGHTS & RECOMMENDATIONS")
        insights_content.append("-" * 40)

        for i, insight in enumerate(current_insights, 1):
            insights_content.append(f"\n{i}. {insight['title']}")
            insights_content.append(f"   Priority: {insight['priority'].upper()}")
            insights_content.append(f"   Metric: {insight['metric']}")
            insights_content.append(f"   Trend: {insight['trend'].upper()}")
            insights_content.append(f"   \n   Summary:")
            insights_content.append(f"   {insight['text']}")
            if 'detailed_analysis' in insight:
                insights_content.append(f"   \n   Detailed Analysis:")
                insights_content.append(f"   {insight['detailed_analysis']}")
            insights_content.append("")

        insights_content.append("📈 ANALYSIS METHODOLOGY")
        insights_content.append("-" * 30)
        insights_content.append("This analysis uses three advanced sentiment analysis methods:")
        insights_content.append("1. TextBlob: Lexicon-based sentiment analysis with polarity and subjectivity scores")
        insights_content.append("2. VADER: Valence Aware Dictionary and Sentiment Reasoner optimized for social media text")
        insights_content.append("3. Custom Rule-Based: Domain-specific keyword analysis with confidence scoring")
        insights_content.append("")
        insights_content.append("The insights are generated using machine learning algorithms that identify")
        insights_content.append("patterns, trends, and actionable recommendations based on the sentiment analysis results.")
        insights_content.append("")
        insights_content.append("🤖 Powered by Advanced AI & Python Data Science")
        insights_content.append("Generated by FeedbackSense AI Platform")

        # Create text file content
        file_content = "\n".join(insights_content)

        # Create a BytesIO object to hold the file
        text_buffer = BytesIO()
        text_buffer.write(file_content.encode('utf-8'))
        text_buffer.seek(0)

        return send_file(
            text_buffer,
            mimetype='text/plain',
            as_attachment=True,
            download_name=f'sentiment_analysis_insights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )

    except Exception as e:
        print(f"Insights download error: {str(e)}")
        return jsonify({'error': 'Failed to create insights download'}), 500

@app.route('/download-dataset')
def download_dataset():
    """Download the analyzed dataset as CSV"""
    global current_analysis_results

    if current_analysis_results is None:
        return jsonify({'error': 'No dataset available for download'}), 404

    try:
        # Create CSV content
        csv_buffer = StringIO()
        current_analysis_results.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Create BytesIO for file download
        file_buffer = BytesIO()
        file_buffer.write(csv_content.encode('utf-8'))
        file_buffer.seek(0)

        return send_file(
            file_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'sentiment_analysis_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

    except Exception as e:
        print(f"Dataset download error: {str(e)}")
        return jsonify({'error': 'Failed to create dataset download'}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Sentiment Analysis Web Application...")
    print("📊 Flask server with Python backend")
    print("🤖 AI-powered sentiment analysis ready")
    print("📁 Chart and insights downloads enabled")
    print("🌐 Open http://localhost:5000 in your browser")
    print("-" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)