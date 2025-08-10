#!/usr/bin/env python3
"""
Cognitive Insights Generator - Advanced EEG Analysis
"""

import numpy as np
import sqlite3
import pickle
import json
from pathlib import Path
from typing import Dict, List
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CognitiveInsightsGenerator:
    def __init__(self, db_paths: List[str]):
        self.db_paths = [Path(path) for path in db_paths]
        self.databases = {}
        self.cognitive_analysis = {}
        self._analyze_cognitive_data()
    
    def _analyze_cognitive_data(self):
        """Load and analyze cognitive data from all databases"""
        for db_path in self.db_paths:
            if not db_path.exists():
                continue
                
            db_name = db_path.stem
            logger.info(f"🧠 Analyzing cognitive data: {db_name}")
            
            cognitive_data = []
            embeddings = []
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(video_embeddings)")
                columns = [col[1] for col in cursor.fetchall()]
                is_fragment_db = 'fragment_id' in columns
                
                if is_fragment_db:
                    query = '''SELECT fragment_id, video_id, fragment_index, start_time_seconds, 
                               embedding_data, eeg_metrics, fragment_duration_seconds
                               FROM video_embeddings ORDER BY video_id, fragment_index'''
                else:
                    query = '''SELECT video_id, video_id, 0, 0, 
                               embedding_data, eeg_metrics, duration_seconds
                               FROM video_embeddings ORDER BY video_id'''
                
                cursor.execute(query)
                for row in cursor.fetchall():
                    fragment_id, video_id, fragment_idx, start_time, embedding_data, eeg_metrics, duration = row
                    
                    embedding = pickle.loads(embedding_data)
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    embeddings.append(embedding)
                    
                    # Parse EEG metrics
                    eeg_dict = json.loads(eeg_metrics) if eeg_metrics else {}
                    attention = eeg_dict.get('attention', {})
                    
                    # Calculate advanced cognitive metrics
                    cognitive_features = {
                        'fragment_id': fragment_id,
                        'video_id': video_id,
                        'video_short': video_id.replace('lsl_stream_20250809_', ''),
                        'fragment_index': fragment_idx,
                        'start_time': start_time,
                        'duration': duration,
                        'focus_level': attention.get('focus_level', 0.5),
                        'engagement': attention.get('engagement', 0.5),
                        'embedding_magnitude': np.linalg.norm(embedding),
                        'embedding_entropy': self._calculate_entropy(embedding),
                        'cognitive_load': self._estimate_cognitive_load(embedding),
                        'attention_stability': self._calculate_stability(embedding),
                        'neural_complexity': np.var(embedding) * (1 + abs(stats.skew(embedding)))
                    }
                    cognitive_data.append(cognitive_features)
            
            # Perform cognitive analysis
            self.databases[db_name] = {
                'cognitive_data': cognitive_data,
                'embeddings': np.array(embeddings),
                'duration': duration if cognitive_data else 0
            }
            
            self._analyze_patterns(db_name)
            logger.info(f"✅ Analyzed {len(cognitive_data)} cognitive samples")
    
    def _calculate_entropy(self, embedding: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        try:
            embedding_range = embedding.max() - embedding.min()
            if embedding_range == 0:
                return 0.0
            normalized = (embedding - embedding.min()) / (embedding_range + 1e-8)
            hist, _ = np.histogram(normalized, bins=20, density=True)
            hist = hist[hist > 0]
            if len(hist) == 0:
                return 0.0
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            return min(entropy, 10.0)  # Cap entropy
        except:
            return 0.5
    
    def _estimate_cognitive_load(self, embedding: np.ndarray) -> float:
        """Estimate cognitive load from embedding characteristics"""
        try:
            magnitude = np.linalg.norm(embedding)
            variance = np.var(embedding)
            entropy = self._calculate_entropy(embedding)
            
            # Normalize and cap values to prevent overflow
            magnitude = min(magnitude / 1e6, 1.0)
            variance = min(variance / 1e6, 1.0)
            entropy = min(entropy / 10.0, 1.0)
            
            load = magnitude * variance * entropy
            return min(max(load, 0.0), 1.0)  # Clamp between 0 and 1
        except:
            return 0.5
    
    def _calculate_stability(self, embedding: np.ndarray) -> float:
        """Calculate attention stability"""
        try:
            variance = np.var(embedding)
            if np.isnan(variance) or np.isinf(variance):
                return 0.5
            # Normalize variance to prevent extreme values
            normalized_var = min(variance / 1e6, 100.0)
            stability = 1.0 / (1.0 + normalized_var)
            return min(max(stability, 0.0), 1.0)
        except:
            return 0.5
    
    def _analyze_patterns(self, db_name: str):
        """Analyze cognitive patterns"""
        cognitive_data = self.databases[db_name]['cognitive_data']
        df = pd.DataFrame(cognitive_data)
        
        # Video-level analysis
        video_analysis = {}
        for video_id in df['video_id'].unique():
            video_data = df[df['video_id'] == video_id]
            
            video_analysis[video_id] = {
                'avg_focus': video_data['focus_level'].mean(),
                'avg_engagement': video_data['engagement'].mean(),
                'avg_cognitive_load': video_data['cognitive_load'].mean(),
                'attention_stability': video_data['attention_stability'].mean(),
                'cognitive_variability': video_data['cognitive_load'].std(),
                'focus_trend': self._calculate_trend(video_data['focus_level'].values),
                'engagement_trend': self._calculate_trend(video_data['engagement'].values),
                'fragment_count': len(video_data),
                'cognitive_profile': self._classify_profile(video_data)
            }
        
        # Clustering analysis
        cognitive_features = df[['focus_level', 'engagement', 'cognitive_load', 'attention_stability']].values
        n_clusters = min(5, len(cognitive_features))
        
        cluster_analysis = {}
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(cognitive_features)
            
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = cognitive_features[cluster_mask]
                
                cluster_analysis[f'cluster_{i}'] = {
                    'size': np.sum(cluster_mask),
                    'avg_focus': np.mean(cluster_data[:, 0]),
                    'avg_engagement': np.mean(cluster_data[:, 1]),
                    'avg_cognitive_load': np.mean(cluster_data[:, 2]),
                    'avg_stability': np.mean(cluster_data[:, 3]),
                    'profile': self._classify_cluster_profile(cluster_data)
                }
        
        self.cognitive_analysis[db_name] = {
            'video_analysis': video_analysis,
            'cluster_analysis': cluster_analysis,
            'global_stats': {
                'avg_focus': df['focus_level'].mean(),
                'avg_engagement': df['engagement'].mean(),
                'avg_cognitive_load': df['cognitive_load'].mean(),
                'focus_std': df['focus_level'].std(),
                'engagement_std': df['engagement'].std(),
                'cognitive_load_std': df['cognitive_load'].std(),
                'total_fragments': len(df),
                'unique_videos': df['video_id'].nunique()
            }
        }
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend slope"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _classify_profile(self, video_data: pd.DataFrame) -> str:
        """Classify cognitive profile"""
        avg_focus = video_data['focus_level'].mean()
        avg_engagement = video_data['engagement'].mean()
        avg_load = video_data['cognitive_load'].mean()
        
        if avg_focus > 0.7 and avg_engagement > 0.7:
            return "High Performance"
        elif avg_focus > 0.6 and avg_load < 0.3:
            return "Focused & Relaxed"
        elif avg_engagement > 0.6 and avg_load > 0.5:
            return "Engaged & Active"
        elif avg_load > 0.7:
            return "High Cognitive Load"
        else:
            return "Moderate State"
    
    def _classify_cluster_profile(self, cluster_data: np.ndarray) -> str:
        """Classify cluster cognitive profile"""
        avg_focus = np.mean(cluster_data[:, 0])
        avg_engagement = np.mean(cluster_data[:, 1])
        avg_load = np.mean(cluster_data[:, 2])
        
        if avg_focus > 0.7 and avg_engagement > 0.7:
            return "Peak Performance"
        elif avg_load > 0.6:
            return "High Mental Load"
        elif avg_focus < 0.4:
            return "Low Attention"
        else:
            return "Balanced State"
    
    def create_insights_dashboard(self, output_path: str = "cognitive_insights_dashboard.html"):
        """Create comprehensive cognitive insights dashboard"""
        logger.info("🎨 Creating cognitive insights dashboard...")
        
        # Get all video IDs
        all_videos = set()
        for db_data in self.databases.values():
            for item in db_data['cognitive_data']:
                all_videos.add(item['video_id'])
        all_videos = sorted(list(all_videos))
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive Insights Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .controls {{ background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; margin-bottom: 20px; }}
        .control-group {{ margin-bottom: 15px; }}
        .control-group label {{ display: block; margin-bottom: 5px; font-weight: 600; }}
        .control-group select {{ width: 100%; padding: 10px; border: 2px solid #e0e0e0; border-radius: 8px; }}
        .viz-section {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; margin-bottom: 25px; }}
        .section-title {{ font-size: 1.5em; font-weight: 600; color: #667eea; margin-bottom: 20px; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; }}
        .viz-container {{ background: #f8f9fa; border-radius: 10px; padding: 20px; }}
        .viz-title {{ font-size: 1.1em; font-weight: 600; margin-bottom: 15px; text-align: center; }}
        .insight-box {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .profile-tag {{ background: #4caf50; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; margin: 2px; display: inline-block; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> Cognitive Insights Dashboard</h1>
            <p>Advanced Analysis of EEG-Derived Cognitive States</p>
            <div style="background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 10px; margin-top: 15px;">
                <strong>🎯 Validation Goal:</strong> Assess whether captured cognitive states provide meaningful insights for neural pattern analysis
            </div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label><i class="fas fa-database"></i> Database:</label>
                <select id="database-select" onchange="updateDashboard()">
                    {chr(10).join(f'<option value="{name}">{self._format_db_name(name)}</option>' for name in self.databases.keys())}
                </select>
            </div>
        </div>
        
        <div id="stats-overview" class="stats-grid">
            <!-- Dynamic stats will be populated here -->
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-chart-scatter"></i> Cognitive State Distribution</div>
            <div class="viz-grid">
                <div class="viz-container">
                    <div class="viz-title">🎯 Focus vs Engagement</div>
                    <div id="focus-engagement-plot"></div>
                </div>
                <div class="viz-container">
                    <div class="viz-title">🧠 Cognitive Load Analysis</div>
                    <div id="cognitive-load-plot"></div>
                </div>
            </div>
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-project-diagram"></i> Pattern Recognition</div>
            <div class="viz-grid">
                <div class="viz-container">
                    <div class="viz-title">🎨 Cognitive Clusters</div>
                    <div id="cluster-plot"></div>
                </div>
                <div class="viz-container">
                    <div class="viz-title">📊 Video Profiles</div>
                    <div id="video-profiles-plot"></div>
                </div>
            </div>
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-microscope"></i> Validation Insights</div>
            <div id="validation-insights">
                <!-- Dynamic insights will be populated here -->
            </div>
        </div>
    </div>

    <script>
        const cognitiveData = {json.dumps({name: {'cognitive_data': db['cognitive_data'], 'analysis': self.cognitive_analysis.get(name, {})} for name, db in self.databases.items()})};
        let currentDb = Object.keys(cognitiveData)[0];
        
        document.addEventListener('DOMContentLoaded', function() {{
            updateDashboard();
        }});
        
        function updateDashboard() {{
            currentDb = document.getElementById('database-select').value;
            updateStats();
            createFocusEngagementPlot();
            createCognitiveLoadPlot();
            createClusterPlot();
            createVideoProfilesPlot();
            updateValidationInsights();
        }}
        
        function updateStats() {{
            const analysis = cognitiveData[currentDb].analysis;
            if (!analysis.global_stats) return;
            
            const stats = analysis.global_stats;
            const statsHtml = `
                <div class="stat-card">
                    <div class="stat-value">${{(stats.avg_focus * 100).toFixed(1)}}%</div>
                    <div>Average Focus</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{(stats.avg_engagement * 100).toFixed(1)}}%</div>
                    <div>Average Engagement</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{stats.avg_cognitive_load.toFixed(3)}}</div>
                    <div>Cognitive Load</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{stats.unique_videos}}</div>
                    <div>Videos Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{stats.total_fragments}}</div>
                    <div>Total Fragments</div>
                </div>
            `;
            document.getElementById('stats-overview').innerHTML = statsHtml;
        }}
        
        function createFocusEngagementPlot() {{
            const data = cognitiveData[currentDb].cognitive_data;
            
            const trace = {{
                x: data.map(d => d.focus_level),
                y: data.map(d => d.engagement),
                mode: 'markers',
                type: 'scatter',
                marker: {{
                    size: data.map(d => d.cognitive_load * 20 + 5),
                    color: data.map(d => d.cognitive_load),
                    colorscale: 'Viridis',
                    showscale: true
                }},
                text: data.map(d => `Video: ${{d.video_short}}<br>Focus: ${{d.focus_level.toFixed(3)}}<br>Engagement: ${{d.engagement.toFixed(3)}}`),
                hovertemplate: '%{{text}}<extra></extra>'
            }};
            
            const layout = {{
                xaxis: {{title: 'Focus Level'}},
                yaxis: {{title: 'Engagement Level'}},
                height: 400
            }};
            
            Plotly.newPlot('focus-engagement-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createCognitiveLoadPlot() {{
            const data = cognitiveData[currentDb].cognitive_data;
            
            const trace = {{
                x: data.map(d => d.cognitive_load),
                type: 'histogram',
                nbinsx: 25,
                marker: {{color: '#667eea', opacity: 0.7}}
            }};
            
            const layout = {{
                xaxis: {{title: 'Cognitive Load'}},
                yaxis: {{title: 'Frequency'}},
                height: 400
            }};
            
            Plotly.newPlot('cognitive-load-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createClusterPlot() {{
            const analysis = cognitiveData[currentDb].analysis;
            if (!analysis.cluster_analysis) return;
            
            const clusters = analysis.cluster_analysis;
            const traces = [];
            
            Object.keys(clusters).forEach((clusterKey, i) => {{
                const cluster = clusters[clusterKey];
                traces.push({{
                    x: [cluster.avg_focus],
                    y: [cluster.avg_engagement],
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{size: cluster.size, opacity: 0.7}},
                    name: `${{cluster.profile}} (n=${{cluster.size}})`,
                    text: `Profile: ${{cluster.profile}}<br>Size: ${{cluster.size}}<br>Focus: ${{cluster.avg_focus.toFixed(3)}}<br>Engagement: ${{cluster.avg_engagement.toFixed(3)}}`,
                    hovertemplate: '%{{text}}<extra></extra>'
                }});
            }});
            
            const layout = {{
                xaxis: {{title: 'Average Focus'}},
                yaxis: {{title: 'Average Engagement'}},
                height: 400
            }};
            
            Plotly.newPlot('cluster-plot', traces, layout, {{responsive: true}});
        }}
        
        function createVideoProfilesPlot() {{
            const analysis = cognitiveData[currentDb].analysis;
            if (!analysis.video_analysis) return;
            
            const videos = analysis.video_analysis;
            const videoIds = Object.keys(videos);
            
            const trace = {{
                x: videoIds.map(id => id.replace('lsl_stream_20250809_', '')),
                y: videoIds.map(id => videos[id].avg_focus),
                type: 'bar',
                marker: {{color: '#667eea'}},
                name: 'Average Focus by Video'
            }};
            
            const layout = {{
                xaxis: {{title: 'Video ID', tickangle: 45}},
                yaxis: {{title: 'Average Focus Level'}},
                height: 400
            }};
            
            Plotly.newPlot('video-profiles-plot', [trace], layout, {{responsive: true}});
        }}
        
        function updateValidationInsights() {{
            const analysis = cognitiveData[currentDb].analysis;
            if (!analysis.global_stats) return;
            
            const stats = analysis.global_stats;
            const focusVariability = stats.focus_std;
            const engagementVariability = stats.engagement_std;
            const loadVariability = stats.cognitive_load_std;
            
            // Calculate validation metrics
            const signalToNoise = stats.avg_focus / focusVariability;
            const stateDiscrimination = focusVariability + engagementVariability + loadVariability;
            const temporalResolution = currentDb.includes('5s') ? 'High' : currentDb.includes('20s') ? 'Medium' : 'Low';
            
            const insights = `
                <div class="insight-box">
                    <h4>📊 Statistical Validity</h4>
                    <p><strong>Signal-to-Noise Ratio:</strong> ${{signalToNoise.toFixed(2)}} (Higher = Better discrimination)</p>
                    <p><strong>State Variability:</strong> ${{stateDiscrimination.toFixed(3)}} (Indicates measurable differences)</p>
                    <p><strong>Temporal Resolution:</strong> ${{temporalResolution}} (${{currentDb.includes('5s') ? '5s' : currentDb.includes('20s') ? '20s' : 'Full'}} fragments)</p>
                </div>
                <div class="insight-box">
                    <h4>🎯 Cognitive Insights</h4>
                    <p><strong>Focus Range:</strong> ${{(stats.avg_focus - stats.focus_std).toFixed(3)}} - ${{(stats.avg_focus + stats.focus_std).toFixed(3)}}</p>
                    <p><strong>Engagement Range:</strong> ${{(stats.avg_engagement - stats.engagement_std).toFixed(3)}} - ${{(stats.avg_engagement + stats.engagement_std).toFixed(3)}}</p>
                    <p><strong>Cognitive Load Spread:</strong> ${{loadVariability.toFixed(3)}} (Variability in mental effort)</p>
                </div>
                <div class="insight-box">
                    <h4>✅ Validation Conclusion</h4>
                    <p><strong>Discriminative Power:</strong> ${{signalToNoise > 2 ? 'High' : signalToNoise > 1 ? 'Moderate' : 'Low'}} - Cognitive states are ${{signalToNoise > 2 ? 'clearly distinguishable' : signalToNoise > 1 ? 'moderately distinguishable' : 'poorly distinguishable'}}</p>
                    <p><strong>Temporal Sensitivity:</strong> ${{temporalResolution}} resolution captures ${{currentDb.includes('5s') ? 'fine-grained' : currentDb.includes('20s') ? 'medium-scale' : 'global'}} cognitive dynamics</p>
                    <p><strong>Practical Utility:</strong> ${{stateDiscrimination > 0.5 ? 'High' : stateDiscrimination > 0.2 ? 'Moderate' : 'Limited'}} - Data shows ${{stateDiscrimination > 0.5 ? 'strong potential' : stateDiscrimination > 0.2 ? 'moderate potential' : 'limited potential'}} for cognitive state analysis</p>
                </div>
            `;
            
            document.getElementById('validation-insights').innerHTML = insights;
        }}
    </script>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Cognitive insights dashboard created: {output_path}")
        return output_path
    
    def _format_db_name(self, name: str) -> str:
        """Format database name for display"""
        if 'video_embeddings_5s' in name:
            return "5s Fragments (High Temporal Resolution)"
        elif 'video_embeddings_20s' in name:
            return "20s Fragments (Medium Temporal Resolution)"
        elif 'video_embeddings' in name:
            return "Full Videos (Global Analysis)"
        return name

def main():
    parser = argparse.ArgumentParser(description="Generate cognitive insights dashboard")
    parser.add_argument("--db_paths", nargs='+', required=True, help="Database paths")
    parser.add_argument("--output", default="cognitive_insights_dashboard.html", help="Output file")
    
    args = parser.parse_args()
    
    print("🧠 Cognitive Insights Dashboard Generator")
    print("=" * 50)
    
    generator = CognitiveInsightsGenerator(args.db_paths)
    output_path = generator.create_insights_dashboard(args.output)
    
    print(f"\n🌐 Open dashboard: file://{Path(output_path).absolute()}")

if __name__ == "__main__":
    main()
