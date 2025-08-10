#!/usr/bin/env python3
"""
Cognitive States for Digital Navigation Analysis
Demonstrates meaningful cognitive state extraction from EEG embeddings
"""

import numpy as np
import sqlite3
import pickle
import json
from pathlib import Path
from typing import Dict, List
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CognitiveNavigationAnalyzer:
    """Extract meaningful cognitive states for digital navigation from EEG embeddings"""
    
    def __init__(self, db_paths: List[str]):
        self.db_paths = [Path(path) for path in db_paths]
        self.cognitive_states = {}
        self.navigation_insights = {}
        self._analyze_cognitive_states()
    
    def _analyze_cognitive_states(self):
        """Extract meaningful cognitive states from EEG embeddings"""
        logger.info("🧠 Extracting meaningful cognitive states...")
        
        for db_path in self.db_paths:
            if not db_path.exists():
                continue
                
            db_name = db_path.stem
            logger.info(f"🔍 Processing: {db_name}")
            
            embeddings = []
            cognitive_features = []
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(video_embeddings)")
                columns = [col[1] for col in cursor.fetchall()]
                is_fragment_db = 'fragment_id' in columns
                
                if is_fragment_db:
                    query = '''SELECT fragment_id, video_id, embedding_data FROM video_embeddings'''
                else:
                    query = '''SELECT video_id, video_id, embedding_data FROM video_embeddings'''
                
                cursor.execute(query)
                for row in cursor.fetchall():
                    fragment_id, video_id, embedding_data = row
                    
                    embedding = pickle.loads(embedding_data)
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    embeddings.append(embedding)
                    
                    # Extract meaningful cognitive states
                    cognitive_state = self._extract_cognitive_features(embedding)
                    cognitive_state.update({
                        'fragment_id': fragment_id,
                        'video_id': video_id,
                        'video_short': video_id.replace('lsl_stream_20250809_', '')
                    })
                    cognitive_features.append(cognitive_state)
            
            self.cognitive_states[db_name] = cognitive_features
            self.navigation_insights[db_name] = self._generate_navigation_insights(cognitive_features)
            
            logger.info(f"✅ Extracted {len(cognitive_features)} cognitive states")
    
    def _extract_cognitive_features(self, embedding: np.ndarray) -> Dict:
        """Extract meaningful cognitive features from embedding"""
        
        # 1. ATTENTION STATES
        attention_focus = 1.0 / (1.0 + np.var(embedding))
        attention_stability = 1.0 - (np.std(embedding) / (np.mean(np.abs(embedding)) + 1e-8))
        
        # 2. COGNITIVE LOAD
        cognitive_load = min(np.linalg.norm(embedding) / 1e6, 1.0)
        mental_capacity = 1.0 - cognitive_load
        
        # 3. ENGAGEMENT LEVEL
        engagement = 1.0 - (np.std(embedding) / (np.mean(np.abs(embedding)) + 1e-8))
        
        # 4. EMOTIONAL STATE
        emotional_valence = (np.mean(embedding) + 1.0) / 2.0  # 0-1 scale
        stress_level = min(np.var(embedding) / 1e6, 1.0)
        
        # 5. DECISION READINESS
        decision_confidence = 1.0 / (1.0 + np.var(embedding) / 1e6)
        uncertainty = np.var(embedding) / (np.mean(embedding**2) + 1e-8)
        
        # 6. DIGITAL NAVIGATION METRICS
        ui_complexity_tolerance = mental_capacity * attention_stability
        interaction_speed_preference = np.std(embedding) / (np.max(np.abs(embedding)) + 1e-8)
        
        return {
            'attention_focus': min(max(attention_focus, 0.0), 1.0),
            'attention_stability': min(max(attention_stability, 0.0), 1.0),
            'cognitive_load': cognitive_load,
            'mental_capacity': mental_capacity,
            'engagement_level': min(max(engagement, 0.0), 1.0),
            'emotional_valence': min(max(emotional_valence, 0.0), 1.0),
            'stress_level': stress_level,
            'decision_confidence': min(max(decision_confidence, 0.0), 1.0),
            'uncertainty': min(max(uncertainty, 0.0), 1.0),
            'ui_complexity_tolerance': min(max(ui_complexity_tolerance, 0.0), 1.0),
            'interaction_speed_preference': min(max(interaction_speed_preference, 0.0), 1.0)
        }
    
    def _generate_navigation_insights(self, cognitive_features: List[Dict]) -> Dict:
        """Generate insights for digital navigation applications"""
        df = pd.DataFrame(cognitive_features)
        
        # User profiles for digital interaction
        profiles = []
        for _, row in df.iterrows():
            if row['attention_focus'] > 0.7 and row['cognitive_load'] < 0.4:
                profile = "Power User"
            elif row['engagement_level'] > 0.6 and row['ui_complexity_tolerance'] > 0.5:
                profile = "Engaged Explorer"
            elif row['stress_level'] > 0.6 or row['cognitive_load'] > 0.7:
                profile = "Needs Support"
            elif row['decision_confidence'] < 0.4 and row['uncertainty'] > 0.6:
                profile = "Uncertain Navigator"
            else:
                profile = "Balanced User"
            profiles.append(profile)
        
        profile_distribution = {}
        for profile in profiles:
            profile_distribution[profile] = profile_distribution.get(profile, 0) + 1
        
        return {
            'avg_attention_focus': df['attention_focus'].mean(),
            'avg_cognitive_load': df['cognitive_load'].mean(),
            'avg_engagement': df['engagement_level'].mean(),
            'avg_stress_level': df['stress_level'].mean(),
            'avg_decision_confidence': df['decision_confidence'].mean(),
            'ui_complexity_tolerance_avg': df['ui_complexity_tolerance'].mean(),
            'user_profiles': profile_distribution,
            'navigation_readiness': (df['decision_confidence'] > 0.7).sum() / len(df) * 100,
            'high_engagement_moments': (df['engagement_level'] > 0.7).sum() / len(df) * 100,
            'support_needed_moments': (df['stress_level'] > 0.6).sum() / len(df) * 100
        }
    
    def create_navigation_dashboard(self, output_path: str = "cognitive_navigation_dashboard.html"):
        """Create cognitive navigation dashboard"""
        logger.info("🎨 Creating cognitive navigation dashboard...")
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive States for Digital Navigation</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .importance-section {{ background: #e8f4fd; border-left: 5px solid #2196f3; padding: 25px; margin: 25px 0; border-radius: 10px; }}
        .application-section {{ background: #f0f8e8; border-left: 5px solid #4caf50; padding: 25px; margin: 25px 0; border-radius: 10px; }}
        .controls {{ background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; margin-bottom: 20px; }}
        .viz-section {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; margin-bottom: 25px; }}
        .section-title {{ font-size: 1.5em; font-weight: 600; color: #667eea; margin-bottom: 20px; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; }}
        .viz-container {{ background: #f8f9fa; border-radius: 10px; padding: 20px; }}
        .viz-title {{ font-size: 1.1em; font-weight: 600; margin-bottom: 15px; text-align: center; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .control-group select {{ width: 100%; padding: 10px; border: 2px solid #e0e0e0; border-radius: 8px; }}
        .application-card {{ background: white; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 4px solid #4caf50; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> Cognitive States for Digital Navigation</h1>
            <p>Meaningful Cognitive Features from EEG Embeddings for Interactive Digital Experiences</p>
        </div>
        
        <div class="importance-section">
            <h2>🎯 The Power of Meaningful Cognitive State Extraction</h2>
            <p><strong>Beyond Traditional Analytics:</strong> This analysis demonstrates how EEG embeddings can extract <em>actionable cognitive states</em> that directly translate to digital interface optimizations.</p>
            
            <h3>🧠 Key Cognitive States for Digital Navigation:</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">🎯</div>
                    <div><strong>Attention Focus</strong><br>UI complexity adaptation</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">⚡</div>
                    <div><strong>Cognitive Load</strong><br>Content complexity optimization</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">❤️</div>
                    <div><strong>Engagement Level</strong><br>Content personalization</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">🤔</div>
                    <div><strong>Decision Confidence</strong><br>Interaction timing</div>
                </div>
            </div>
        </div>
        
        <div class="application-section">
            <h2>🚀 Digital Navigation Applications</h2>
            
            <div class="application-card">
                <h4>🎨 Adaptive User Interfaces</h4>
                <p><strong>Real-time UI Adaptation:</strong> Simplify interfaces when cognitive load is high, increase complexity when attention is focused.</p>
                <p><strong>Example:</strong> E-commerce checkout simplifies when stress is detected, adds product comparisons when engagement is high.</p>
            </div>
            
            <div class="application-card">
                <h4>📱 Personalized Content Delivery</h4>
                <p><strong>Engagement-Based Content:</strong> Adjust content complexity and presentation based on real-time engagement levels.</p>
                <p><strong>Example:</strong> Educational platform presents bite-sized content when cognitive fatigue is detected.</p>
            </div>
            
            <div class="application-card">
                <h4>⏰ Optimal Interaction Timing</h4>
                <p><strong>Decision Support Timing:</strong> Present choices when decision confidence is high, provide guidance when uncertainty peaks.</p>
                <p><strong>Example:</strong> Investment app suggests changes only when user shows high decision confidence.</p>
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label><i class="fas fa-database"></i> Database:</label>
                <select id="database-select" onchange="updateAnalysis()">
                    {chr(10).join(f'<option value="{db}">{self._format_db_name(db)}</option>' for db in self.cognitive_states.keys())}
                </select>
            </div>
        </div>
        
        <div id="insights-overview" class="metrics-grid">
            <!-- Dynamic insights will be populated here -->
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-chart-line"></i> Cognitive State Analysis</div>
            <div class="viz-grid">
                <div class="viz-container">
                    <div class="viz-title">🎯 Attention vs Cognitive Load</div>
                    <div id="attention-load-plot"></div>
                </div>
                <div class="viz-container">
                    <div class="viz-title">❤️ Engagement vs Decision Confidence</div>
                    <div id="engagement-confidence-plot"></div>
                </div>
            </div>
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-users"></i> Digital Interaction Profiles</div>
            <div class="viz-grid">
                <div class="viz-container">
                    <div class="viz-title">👥 User Profile Distribution</div>
                    <div id="user-profiles-plot"></div>
                </div>
                <div class="viz-container">
                    <div class="viz-title">📊 Navigation Readiness Over Time</div>
                    <div id="navigation-readiness-plot"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const cognitiveData = {json.dumps(self.cognitive_states, indent=2, default=str)};
        const navigationInsights = {json.dumps(self.navigation_insights, indent=2, default=str)};
        let currentDb = Object.keys(cognitiveData)[0];
        
        document.addEventListener('DOMContentLoaded', function() {{
            updateAnalysis();
        }});
        
        function updateAnalysis() {{
            currentDb = document.getElementById('database-select').value;
            updateInsightsOverview();
            createAttentionLoadPlot();
            createEngagementConfidencePlot();
            createUserProfilesPlot();
            createNavigationReadinessPlot();
        }}
        
        function updateInsightsOverview() {{
            const insights = navigationInsights[currentDb];
            
            const insightsHtml = `
                <div class="metric-card">
                    <div class="metric-value">${{(insights.avg_attention_focus * 100).toFixed(1)}}%</div>
                    <div>Average Attention Focus</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{(insights.avg_cognitive_load * 100).toFixed(1)}}%</div>
                    <div>Average Cognitive Load</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{(insights.avg_engagement * 100).toFixed(1)}}%</div>
                    <div>Average Engagement</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{insights.navigation_readiness.toFixed(1)}}%</div>
                    <div>Navigation Readiness</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{insights.high_engagement_moments.toFixed(1)}}%</div>
                    <div>High Engagement Moments</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{insights.support_needed_moments.toFixed(1)}}%</div>
                    <div>Support Needed Moments</div>
                </div>
            `;
            document.getElementById('insights-overview').innerHTML = insightsHtml;
        }}
        
        function createAttentionLoadPlot() {{
            const data = cognitiveData[currentDb];
            
            const trace = {{
                x: data.map(d => d.attention_focus),
                y: data.map(d => d.cognitive_load),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: data.map(d => d.engagement_level * 20 + 5),
                    color: data.map(d => d.stress_level),
                    colorscale: 'RdYlBu_r',
                    showscale: true,
                    colorbar: {{title: 'Stress Level'}}
                }},
                text: data.map(d => `Video: ${{d.video_short}}<br>Attention: ${{d.attention_focus.toFixed(3)}}<br>Load: ${{d.cognitive_load.toFixed(3)}}<br>Stress: ${{d.stress_level.toFixed(3)}}`),
                hovertemplate: '%{{text}}<extra></extra>'
            }};
            
            const layout = {{
                xaxis: {{title: 'Attention Focus', range: [0, 1]}},
                yaxis: {{title: 'Cognitive Load', range: [0, 1]}},
                height: 400
            }};
            
            Plotly.newPlot('attention-load-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createEngagementConfidencePlot() {{
            const data = cognitiveData[currentDb];
            
            const trace = {{
                x: data.map(d => d.engagement_level),
                y: data.map(d => d.decision_confidence),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: data.map(d => d.ui_complexity_tolerance * 20 + 5),
                    color: data.map(d => d.emotional_valence),
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{title: 'Emotional Valence'}}
                }},
                text: data.map(d => `Video: ${{d.video_short}}<br>Engagement: ${{d.engagement_level.toFixed(3)}}<br>Confidence: ${{d.decision_confidence.toFixed(3)}}`),
                hovertemplate: '%{{text}}<extra></extra>'
            }};
            
            const layout = {{
                xaxis: {{title: 'Engagement Level', range: [0, 1]}},
                yaxis: {{title: 'Decision Confidence', range: [0, 1]}},
                height: 400
            }};
            
            Plotly.newPlot('engagement-confidence-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createUserProfilesPlot() {{
            const insights = navigationInsights[currentDb];
            const profiles = insights.user_profiles;
            
            const trace = {{
                labels: Object.keys(profiles),
                values: Object.values(profiles),
                type: 'pie',
                marker: {{
                    colors: ['#4caf50', '#2196f3', '#ff9800', '#f44336', '#9c27b0']
                }}
            }};
            
            const layout = {{
                height: 400,
                showlegend: true
            }};
            
            Plotly.newPlot('user-profiles-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createNavigationReadinessPlot() {{
            const data = cognitiveData[currentDb];
            
            const readinessScores = data.map((d, i) => {{
                return {{
                    x: i,
                    y: (d.attention_focus * 0.3 + (1 - d.cognitive_load) * 0.3 + d.decision_confidence * 0.4),
                    video: d.video_short
                }};
            }});
            
            const trace = {{
                x: readinessScores.map(r => r.x),
                y: readinessScores.map(r => r.y),
                type: 'scatter',
                mode: 'lines+markers',
                marker: {{color: '#667eea'}},
                name: 'Navigation Readiness'
            }};
            
            const layout = {{
                xaxis: {{title: 'Fragment Index'}},
                yaxis: {{title: 'Navigation Readiness Score', range: [0, 1]}},
                height: 400
            }};
            
            Plotly.newPlot('navigation-readiness-plot', [trace], layout, {{responsive: true}});
        }}
    </script>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Cognitive navigation dashboard created: {output_path}")
        return output_path
    
    def _format_db_name(self, db_name: str) -> str:
        """Format database name for display"""
        if '5s' in db_name:
            return "5s Fragments (High Resolution)"
        elif '20s' in db_name:
            return "20s Fragments (Medium Resolution)"
        else:
            return "Full Videos (Global Analysis)"

def main():
    parser = argparse.ArgumentParser(description="Generate cognitive navigation dashboard")
    parser.add_argument("--db_paths", nargs='+', required=True, help="Database paths")
    parser.add_argument("--output", default="cognitive_navigation_dashboard.html", help="Output file")
    
    args = parser.parse_args()
    
    print("🧠 Cognitive States for Digital Navigation Analyzer")
    print("=" * 60)
    
    analyzer = CognitiveNavigationAnalyzer(args.db_paths)
    output_path = analyzer.create_navigation_dashboard(args.output)
    
    print(f"\n🌐 Open dashboard: file://{Path(output_path).absolute()}")

if __name__ == "__main__":
    main()
