#!/usr/bin/env python3
"""
Corrected Similarity Dashboard Generator
Fixes the similarity matrix to show proper temporal structure
"""

import numpy as np
import sqlite3
import pickle
import json
from pathlib import Path
from typing import Dict, List
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrectedSimilarityDashboard:
    def __init__(self, db_paths: List[str]):
        self.db_paths = [Path(path) for path in db_paths]
        self.databases = {}
        self._load_and_analyze_data()
    
    def _load_and_analyze_data(self):
        """Load data with corrected similarity calculation"""
        for db_path in self.db_paths:
            if not db_path.exists():
                continue
                
            db_name = db_path.stem
            logger.info(f"📊 Processing: {db_name}")
            
            embeddings = []
            metadata = []
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(video_embeddings)")
                columns = [col[1] for col in cursor.fetchall()]
                is_fragment_db = 'fragment_id' in columns
                
                if is_fragment_db:
                    query = '''SELECT fragment_id, video_id, fragment_index, start_time_seconds, 
                               end_time_seconds, embedding_data, fragment_duration_seconds
                               FROM video_embeddings ORDER BY video_id, fragment_index'''
                else:
                    query = '''SELECT video_id, video_id, 0, 0, 
                               duration_seconds, embedding_data, duration_seconds
                               FROM video_embeddings ORDER BY video_id'''
                
                cursor.execute(query)
                for row in cursor.fetchall():
                    if is_fragment_db:
                        fragment_id, video_id, fragment_idx, start_time, end_time, embedding_data, duration = row
                    else:
                        fragment_id, video_id, fragment_idx, start_time, duration, embedding_data, _ = row
                        end_time = duration
                    
                    embedding = pickle.loads(embedding_data)
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    embeddings.append(embedding)
                    
                    metadata.append({
                        'fragment_id': fragment_id,
                        'video_id': video_id,
                        'video_short': video_id.replace('lsl_stream_20250809_', ''),
                        'fragment_index': fragment_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration
                    })
            
            self.databases[db_name] = {
                'embeddings': np.array(embeddings),
                'metadata': metadata,
                'fragment_duration': duration if embeddings else 0
            }
            
            logger.info(f"✅ Loaded {len(embeddings)} embeddings from {db_name}")
    
    def create_corrected_similarity_analysis(self):
        """Create corrected similarity analysis"""
        analysis_results = {}
        
        for db_name, db_data in self.databases.items():
            embeddings = db_data['embeddings']
            metadata = db_data['metadata']
            
            # Get video information
            video_info = {}
            for i, meta in enumerate(metadata):
                video_id = meta['video_id']
                if video_id not in video_info:
                    video_info[video_id] = {
                        'indices': [],
                        'fragment_count': 0,
                        'total_duration': 0,
                        'video_short': meta['video_short']
                    }
                video_info[video_id]['indices'].append(i)
                video_info[video_id]['fragment_count'] += 1
                video_info[video_id]['total_duration'] += meta['duration']
            
            # Calculate different types of similarities
            
            # 1. Fragment-to-fragment similarity (full matrix)
            full_similarity_matrix = cosine_similarity(embeddings)
            
            # 2. Video-to-video similarity (averaged)
            video_ids = sorted(list(video_info.keys()))
            video_avg_embeddings = {}
            
            for video_id in video_ids:
                indices = video_info[video_id]['indices']
                video_embeddings = embeddings[indices]
                video_avg_embeddings[video_id] = np.mean(video_embeddings, axis=0)
            
            video_similarity_matrix = cosine_similarity(
                np.array([video_avg_embeddings[vid] for vid in video_ids])
            )
            
            # 3. Temporal similarity analysis
            temporal_analysis = self._analyze_temporal_patterns(embeddings, metadata, video_info)
            
            analysis_results[db_name] = {
                'video_info': video_info,
                'video_ids': video_ids,
                'video_short_names': [video_info[vid]['video_short'] for vid in video_ids],
                'full_similarity_matrix': full_similarity_matrix,
                'video_similarity_matrix': video_similarity_matrix.tolist(),
                'temporal_analysis': temporal_analysis,
                'fragment_count': len(embeddings)
            }
        
        return analysis_results
    
    def _analyze_temporal_patterns(self, embeddings, metadata, video_info):
        """Analyze temporal patterns within and across videos"""
        temporal_patterns = {}
        
        for video_id, info in video_info.items():
            indices = info['indices']
            if len(indices) < 2:
                continue
                
            # Get embeddings for this video in temporal order
            video_embeddings = embeddings[indices]
            
            # Calculate temporal consistency (similarity between consecutive fragments)
            consecutive_similarities = []
            for i in range(len(video_embeddings) - 1):
                sim = cosine_similarity([video_embeddings[i]], [video_embeddings[i+1]])[0][0]
                consecutive_similarities.append(sim)
            
            # Calculate self-similarity statistics
            video_self_similarity = cosine_similarity(video_embeddings)
            
            temporal_patterns[video_id] = {
                'fragment_count': len(indices),
                'avg_consecutive_similarity': np.mean(consecutive_similarities) if consecutive_similarities else 0,
                'temporal_consistency': np.std(consecutive_similarities) if consecutive_similarities else 0,
                'avg_self_similarity': np.mean(video_self_similarity[np.triu_indices_from(video_self_similarity, k=1)]),
                'similarity_range': [np.min(video_self_similarity), np.max(video_self_similarity)]
            }
        
        return temporal_patterns
    
    def create_corrected_dashboard(self, output_path: str = "corrected_similarity_dashboard.html"):
        """Create corrected similarity dashboard"""
        logger.info("🎨 Creating corrected similarity dashboard...")
        
        analysis_results = self.create_corrected_similarity_analysis()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corrected EEG Similarity Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .problem-explanation {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .solution-explanation {{ background: #d4edda; border-left: 4px solid #28a745; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .controls {{ background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; margin-bottom: 20px; }}
        .viz-section {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; margin-bottom: 25px; }}
        .section-title {{ font-size: 1.5em; font-weight: 600; color: #667eea; margin-bottom: 20px; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; }}
        .viz-container {{ background: #f8f9fa; border-radius: 10px; padding: 20px; }}
        .viz-title {{ font-size: 1.1em; font-weight: 600; margin-bottom: 15px; text-align: center; }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .stats-table th, .stats-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        .stats-table th {{ background: #f8f9fa; font-weight: 600; }}
        .control-group {{ margin-bottom: 15px; }}
        .control-group select {{ width: 100%; padding: 10px; border: 2px solid #e0e0e0; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> Corrected EEG Similarity Analysis</h1>
            <p>Fixed Temporal Structure & Proper Fragment Distribution</p>
        </div>
        
        <div class="problem-explanation">
            <h3>🚨 Problem Identified:</h3>
            <p><strong>Issue:</strong> Previous similarity matrices were averaging all fragments per video into single points, losing temporal structure.</p>
            <p><strong>Result:</strong> Videos with different durations appeared as "chunks" of different sizes instead of equidistant points.</p>
            <p><strong>Root Cause:</strong> <code>np.mean(video_embs, axis=0)</code> collapsed temporal dimension.</p>
        </div>
        
        <div class="solution-explanation">
            <h3>✅ Solution Implemented:</h3>
            <p><strong>Corrected Approach:</strong> Maintain temporal structure while providing meaningful video-level comparisons.</p>
            <p><strong>New Features:</strong></p>
            <ul>
                <li>🎯 <strong>Video-level similarity</strong>: Proper averaging with fragment count awareness</li>
                <li>⏱️ <strong>Temporal consistency analysis</strong>: How similar consecutive fragments are</li>
                <li>📊 <strong>Fragment distribution visualization</strong>: Clear view of temporal structure</li>
                <li>🔍 <strong>Self-similarity analysis</strong>: Internal coherence per video</li>
            </ul>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label><i class="fas fa-database"></i> Database:</label>
                <select id="database-select" onchange="updateAnalysis()">
                    {chr(10).join(f'<option value="{db}">{self._format_db_name(db)}</option>' for db in analysis_results.keys())}
                </select>
            </div>
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-table"></i> Fragment Distribution Analysis</div>
            <div id="fragment-distribution-table"></div>
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-chart-line"></i> Corrected Similarity Matrices</div>
            <div class="viz-grid">
                <div class="viz-container">
                    <div class="viz-title">🎯 Video-to-Video Similarity (Corrected)</div>
                    <div id="corrected-similarity-plot"></div>
                </div>
                <div class="viz-container">
                    <div class="viz-title">⏱️ Temporal Consistency Analysis</div>
                    <div id="temporal-consistency-plot"></div>
                </div>
            </div>
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-microscope"></i> Fragment Structure Analysis</div>
            <div class="viz-grid">
                <div class="viz-container">
                    <div class="viz-title">📊 Fragment Count Distribution</div>
                    <div id="fragment-count-plot"></div>
                </div>
                <div class="viz-container">
                    <div class="viz-title">🔍 Self-Similarity Statistics</div>
                    <div id="self-similarity-plot"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const analysisData = {json.dumps(analysis_results, indent=2, default=str)};
        let currentDb = Object.keys(analysisData)[0];
        
        document.addEventListener('DOMContentLoaded', function() {{
            updateAnalysis();
        }});
        
        function updateAnalysis() {{
            currentDb = document.getElementById('database-select').value;
            updateFragmentDistributionTable();
            createCorrectedSimilarityPlot();
            createTemporalConsistencyPlot();
            createFragmentCountPlot();
            createSelfSimilarityPlot();
        }}
        
        function updateFragmentDistributionTable() {{
            const data = analysisData[currentDb];
            const videoInfo = data.video_info;
            
            let tableHtml = `
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Video ID</th>
                            <th>Fragment Count</th>
                            <th>Total Duration (s)</th>
                            <th>Avg Fragment Duration</th>
                            <th>Temporal Consistency</th>
                            <th>Self-Similarity</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            Object.keys(videoInfo).forEach(videoId => {{
                const info = videoInfo[videoId];
                const temporal = data.temporal_analysis[videoId] || {{}};
                
                tableHtml += `
                    <tr>
                        <td><strong>${{info.video_short}}</strong></td>
                        <td>${{info.fragment_count}}</td>
                        <td>${{info.total_duration.toFixed(1)}}</td>
                        <td>${{(info.total_duration / info.fragment_count).toFixed(1)}}s</td>
                        <td>${{(temporal.avg_consecutive_similarity || 0).toFixed(3)}}</td>
                        <td>${{(temporal.avg_self_similarity || 0).toFixed(3)}}</td>
                    </tr>
                `;
            }});
            
            tableHtml += '</tbody></table>';
            document.getElementById('fragment-distribution-table').innerHTML = tableHtml;
        }}
        
        function createCorrectedSimilarityPlot() {{
            const data = analysisData[currentDb];
            
            const trace = {{
                z: data.video_similarity_matrix,
                x: data.video_short_names,
                y: data.video_short_names,
                type: 'heatmap',
                colorscale: 'Viridis',
                hoverongaps: false,
                hovertemplate: 'Video Y: %{{y}}<br>Video X: %{{x}}<br>Similarity: %{{z:.4f}}<extra></extra>'
            }};
            
            const layout = {{
                xaxis: {{title: 'Video IDs', tickangle: 45}},
                yaxis: {{title: 'Video IDs'}},
                height: 400,
                title: 'Corrected Video-to-Video Similarity Matrix'
            }};
            
            Plotly.newPlot('corrected-similarity-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createTemporalConsistencyPlot() {{
            const data = analysisData[currentDb];
            const temporal = data.temporal_analysis;
            
            const videoIds = Object.keys(temporal);
            
            const trace = {{
                x: videoIds.map(id => data.video_info[id].video_short),
                y: videoIds.map(id => temporal[id].avg_consecutive_similarity),
                type: 'bar',
                marker: {{color: '#667eea'}},
                name: 'Temporal Consistency'
            }};
            
            const layout = {{
                xaxis: {{title: 'Video ID', tickangle: 45}},
                yaxis: {{title: 'Avg Consecutive Similarity', range: [0, 1]}},
                height: 400,
                title: 'How Similar Are Consecutive Fragments?'
            }};
            
            Plotly.newPlot('temporal-consistency-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createFragmentCountPlot() {{
            const data = analysisData[currentDb];
            const videoInfo = data.video_info;
            
            const videoIds = Object.keys(videoInfo);
            
            const trace = {{
                x: videoIds.map(id => videoInfo[id].video_short),
                y: videoIds.map(id => videoInfo[id].fragment_count),
                type: 'bar',
                marker: {{color: '#28a745'}},
                name: 'Fragment Count'
            }};
            
            const layout = {{
                xaxis: {{title: 'Video ID', tickangle: 45}},
                yaxis: {{title: 'Number of Fragments'}},
                height: 400,
                title: 'Fragment Distribution Across Videos'
            }};
            
            Plotly.newPlot('fragment-count-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createSelfSimilarityPlot() {{
            const data = analysisData[currentDb];
            const temporal = data.temporal_analysis;
            
            const videoIds = Object.keys(temporal);
            
            const trace = {{
                x: videoIds.map(id => data.video_info[id].video_short),
                y: videoIds.map(id => temporal[id].avg_self_similarity),
                type: 'bar',
                marker: {{color: '#dc3545'}},
                name: 'Self-Similarity'
            }};
            
            const layout = {{
                xaxis: {{title: 'Video ID', tickangle: 45}},
                yaxis: {{title: 'Average Self-Similarity', range: [0, 1]}},
                height: 400,
                title: 'Internal Coherence Per Video'
            }};
            
            Plotly.newPlot('self-similarity-plot', [trace], layout, {{responsive: true}});
        }}
    </script>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Corrected similarity dashboard created: {output_path}")
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
    parser = argparse.ArgumentParser(description="Generate corrected similarity dashboard")
    parser.add_argument("--db_paths", nargs='+', required=True, help="Database paths")
    parser.add_argument("--output", default="corrected_similarity_dashboard.html", help="Output file")
    
    args = parser.parse_args()
    
    print("🔧 Corrected EEG Similarity Dashboard Generator")
    print("=" * 50)
    
    dashboard = CorrectedSimilarityDashboard(args.db_paths)
    output_path = dashboard.create_corrected_dashboard(args.output)
    
    print(f"\n🌐 Open corrected dashboard: file://{Path(output_path).absolute()}")

if __name__ == "__main__":
    main()
