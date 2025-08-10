#!/usr/bin/env python3
"""
Meaningful Similarity Analysis - Corrected Version
Fixes conceptual issues with video ID treatment and similarity interpretation
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

class MeaningfulSimilarityAnalyzer:
    def __init__(self, db_paths: List[str]):
        self.db_paths = [Path(path) for path in db_paths]
        self.databases = {}
        self._load_data()
    
    def _load_data(self):
        """Load data with proper categorical treatment of video IDs"""
        for db_path in self.db_paths:
            if not db_path.exists():
                continue
                
            db_name = db_path.stem
            logger.info(f"📊 Loading: {db_name}")
            
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
                        'duration': duration,
                        'is_fragment_db': is_fragment_db
                    })
            
            self.databases[db_name] = {
                'embeddings': np.array(embeddings),
                'metadata': metadata,
                'is_fragment_db': is_fragment_db,
                'fragment_duration': duration if embeddings else 0
            }
            
            logger.info(f"✅ Loaded {len(embeddings)} embeddings")
    
    def analyze_meaningful_similarities(self):
        """Analyze similarities with proper interpretation"""
        analysis_results = {}
        
        for db_name, db_data in self.databases.items():
            embeddings = db_data['embeddings']
            metadata = db_data['metadata']
            is_fragment_db = db_data['is_fragment_db']
            
            if is_fragment_db:
                # Fragment database: analyze temporal and cross-video patterns
                analysis = self._analyze_fragment_similarities(embeddings, metadata, db_name)
            else:
                # Full video database: analyze video-level patterns
                analysis = self._analyze_video_similarities(embeddings, metadata, db_name)
            
            analysis_results[db_name] = analysis
        
        return analysis_results
    
    def _analyze_fragment_similarities(self, embeddings, metadata, db_name):
        """Analyze fragment-level similarities with meaningful interpretation"""
        
        # Group by video
        video_groups = {}
        for i, meta in enumerate(metadata):
            video_id = meta['video_id']
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(i)
        
        # 1. Intra-video similarity (temporal consistency within videos)
        intra_video_similarities = {}
        for video_id, indices in video_groups.items():
            if len(indices) > 1:
                video_embeddings = embeddings[indices]
                similarity_matrix = cosine_similarity(video_embeddings)
                
                # Calculate temporal consistency (consecutive fragments)
                consecutive_sims = []
                for i in range(len(similarity_matrix) - 1):
                    consecutive_sims.append(similarity_matrix[i, i+1])
                
                intra_video_similarities[video_id] = {
                    'avg_consecutive_similarity': np.mean(consecutive_sims) if consecutive_sims else 0,
                    'avg_overall_similarity': np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
                    'similarity_std': np.std(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
                    'fragment_count': len(indices)
                }
        
        # 2. Inter-video similarity (cross-video patterns)
        video_representatives = {}
        for video_id, indices in video_groups.items():
            video_embeddings = embeddings[indices]
            video_representatives[video_id] = np.mean(video_embeddings, axis=0)
        
        video_ids = list(video_representatives.keys())
        video_matrix = np.array([video_representatives[vid] for vid in video_ids])
        inter_video_similarity = cosine_similarity(video_matrix)
        
        return {
            'type': 'fragment_analysis',
            'video_ids': [vid.replace('lsl_stream_20250809_', '') for vid in video_ids],
            'intra_video_similarities': intra_video_similarities,
            'inter_video_similarity_matrix': inter_video_similarity.tolist(),
            'interpretation': {
                'intra_video': 'Temporal consistency within each video (how similar consecutive fragments are)',
                'inter_video': 'Cross-video similarity (how similar different videos are overall)',
                'meaningful_because': f'Fragments allow temporal analysis within videos and comparison across videos'
            }
        }
    
    def _analyze_video_similarities(self, embeddings, metadata, db_name):
        """Analyze full video similarities with proper interpretation"""
        
        # For full videos, we have one embedding per video
        video_ids = [meta['video_short'] for meta in metadata]
        
        if len(embeddings) == 1:
            return {
                'type': 'single_video',
                'video_id': video_ids[0],
                'interpretation': {
                    'similarity': 'Cannot compute similarity with only one video',
                    'meaningful_analysis': 'Need multiple videos for similarity comparison',
                    'alternative': 'Analyze embedding characteristics instead of similarity'
                },
                'embedding_analysis': self._analyze_single_embedding(embeddings[0])
            }
        
        # Multiple videos: compute pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        return {
            'type': 'multi_video',
            'video_ids': video_ids,
            'similarity_matrix': similarity_matrix.tolist(),
            'interpretation': {
                'similarity': 'Global similarity between complete videos',
                'meaningful_because': 'Each video represented by single embedding capturing overall neural pattern',
                'diagonal_values': 'Always 1.0 (video similar to itself)',
                'off_diagonal': 'Cross-video similarities showing content/cognitive pattern overlap'
            }
        }
    
    def _analyze_single_embedding(self, embedding):
        """Analyze characteristics of a single embedding"""
        return {
            'magnitude': float(np.linalg.norm(embedding)),
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'min': float(np.min(embedding)),
            'max': float(np.max(embedding)),
            'sparsity': float(np.sum(np.abs(embedding) < 1e-6) / len(embedding)),
            'interpretation': 'Embedding characteristics analysis instead of similarity'
        }
    
    def create_meaningful_dashboard(self, output_path: str = "meaningful_similarity_dashboard.html"):
        """Create dashboard with corrected similarity interpretation"""
        logger.info("🎨 Creating meaningful similarity dashboard...")
        
        analysis_results = self.analyze_meaningful_similarities()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meaningful EEG Similarity Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .correction-section {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .interpretation-section {{ background: #d4edda; border-left: 4px solid #28a745; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .controls {{ background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; margin-bottom: 20px; }}
        .viz-section {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; margin-bottom: 25px; }}
        .section-title {{ font-size: 1.5em; font-weight: 600; color: #667eea; margin-bottom: 20px; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; }}
        .viz-container {{ background: #f8f9fa; border-radius: 10px; padding: 20px; }}
        .viz-title {{ font-size: 1.1em; font-weight: 600; margin-bottom: 15px; text-align: center; }}
        .control-group select {{ width: 100%; padding: 10px; border: 2px solid #e0e0e0; border-radius: 8px; }}
        .interpretation-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> Meaningful EEG Similarity Analysis</h1>
            <p>Corrected Interpretation of Video IDs and Similarity Metrics</p>
        </div>
        
        <div class="correction-section">
            <h3>🔧 Corrections Applied:</h3>
            <p><strong>1. Video IDs as Categorical Labels:</strong> Video IDs (175k, 160k, etc.) are now treated as categorical labels, not numeric values.</p>
            <p><strong>2. Clear Similarity Definition:</strong> Explicitly defined what similarity means for each database type.</p>
            <p><strong>3. Meaningful Interpretations:</strong> Added clear explanations of what each similarity metric represents.</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label><i class="fas fa-database"></i> Database:</label>
                <select id="database-select" onchange="updateAnalysis()">
                    {chr(10).join(f'<option value="{db}">{self._format_db_name(db)}</option>' for db in analysis_results.keys())}
                </select>
            </div>
        </div>
        
        <div id="interpretation-section" class="interpretation-section">
            <!-- Dynamic interpretation will be populated here -->
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-chart-line"></i> Similarity Analysis</div>
            <div id="similarity-visualization">
                <!-- Dynamic visualizations will be populated here -->
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
            updateInterpretation();
            createVisualization();
        }}
        
        function updateInterpretation() {{
            const data = analysisData[currentDb];
            const interpretation = data.interpretation || {{}};
            
            let interpretationHtml = `
                <h3>📊 What This Analysis Means:</h3>
                <div class="interpretation-box">
                    <h4>Analysis Type: ${{data.type}}</h4>
            `;
            
            if (data.type === 'fragment_analysis') {{
                interpretationHtml += `
                    <p><strong>Intra-Video Similarity:</strong> ${{interpretation.intra_video}}</p>
                    <p><strong>Inter-Video Similarity:</strong> ${{interpretation.inter_video}}</p>
                    <p><strong>Why Meaningful:</strong> ${{interpretation.meaningful_because}}</p>
                `;
            }} else if (data.type === 'single_video') {{
                interpretationHtml += `
                    <p><strong>Limitation:</strong> ${{interpretation.similarity}}</p>
                    <p><strong>Alternative Analysis:</strong> ${{interpretation.alternative}}</p>
                `;
            }} else if (data.type === 'multi_video') {{
                interpretationHtml += `
                    <p><strong>Similarity Meaning:</strong> ${{interpretation.similarity}}</p>
                    <p><strong>Why Meaningful:</strong> ${{interpretation.meaningful_because}}</p>
                    <p><strong>Diagonal Values:</strong> ${{interpretation.diagonal_values}}</p>
                    <p><strong>Off-Diagonal Values:</strong> ${{interpretation.off_diagonal}}</p>
                `;
            }}
            
            interpretationHtml += '</div>';
            document.getElementById('interpretation-section').innerHTML = interpretationHtml;
        }}
        
        function createVisualization() {{
            const data = analysisData[currentDb];
            
            if (data.type === 'fragment_analysis') {{
                createFragmentAnalysisViz(data);
            }} else if (data.type === 'single_video') {{
                createSingleVideoViz(data);
            }} else if (data.type === 'multi_video') {{
                createMultiVideoViz(data);
            }}
        }}
        
        function createFragmentAnalysisViz(data) {{
            const vizHtml = `
                <div class="viz-grid">
                    <div class="viz-container">
                        <div class="viz-title">🎯 Inter-Video Similarity Matrix</div>
                        <div id="inter-video-plot"></div>
                    </div>
                    <div class="viz-container">
                        <div class="viz-title">⏱️ Temporal Consistency Within Videos</div>
                        <div id="temporal-consistency-plot"></div>
                    </div>
                </div>
            `;
            document.getElementById('similarity-visualization').innerHTML = vizHtml;
            
            // Inter-video similarity heatmap
            const trace1 = {{
                z: data.inter_video_similarity_matrix,
                x: data.video_ids,
                y: data.video_ids,
                type: 'heatmap',
                colorscale: 'Viridis',
                hoverongaps: false
            }};
            
            Plotly.newPlot('inter-video-plot', [trace1], {{
                xaxis: {{title: 'Video IDs (Categorical)'}},
                yaxis: {{title: 'Video IDs (Categorical)'}},
                height: 400
            }}, {{responsive: true}});
            
            // Temporal consistency bar chart
            const intraVideoData = data.intra_video_similarities;
            const videoIds = Object.keys(intraVideoData);
            
            const trace2 = {{
                x: videoIds.map(id => id.replace('lsl_stream_20250809_', '')),
                y: videoIds.map(id => intraVideoData[id].avg_consecutive_similarity),
                type: 'bar',
                marker: {{color: '#667eea'}}
            }};
            
            Plotly.newPlot('temporal-consistency-plot', [trace2], {{
                xaxis: {{title: 'Video IDs (Categorical)'}},
                yaxis: {{title: 'Temporal Consistency', range: [0, 1]}},
                height: 400
            }}, {{responsive: true}});
        }}
        
        function createSingleVideoViz(data) {{
            const vizHtml = `
                <div class="viz-container">
                    <div class="viz-title">📊 Single Video Embedding Analysis</div>
                    <div id="single-video-plot"></div>
                    <p><strong>Note:</strong> Cannot compute similarity with only one video. Showing embedding characteristics instead.</p>
                </div>
            `;
            document.getElementById('similarity-visualization').innerHTML = vizHtml;
            
            const analysis = data.embedding_analysis;
            const trace = {{
                x: ['Magnitude', 'Mean', 'Std', 'Min', 'Max', 'Sparsity'],
                y: [analysis.magnitude, analysis.mean, analysis.std, analysis.min, analysis.max, analysis.sparsity],
                type: 'bar',
                marker: {{color: '#28a745'}}
            }};
            
            Plotly.newPlot('single-video-plot', [trace], {{
                xaxis: {{title: 'Embedding Characteristics'}},
                yaxis: {{title: 'Values'}},
                height: 400
            }}, {{responsive: true}});
        }}
        
        function createMultiVideoViz(data) {{
            const vizHtml = `
                <div class="viz-container">
                    <div class="viz-title">🎯 Video-to-Video Similarity Matrix</div>
                    <div id="multi-video-plot"></div>
                </div>
            `;
            document.getElementById('similarity-visualization').innerHTML = vizHtml;
            
            const trace = {{
                z: data.similarity_matrix,
                x: data.video_ids,
                y: data.video_ids,
                type: 'heatmap',
                colorscale: 'Viridis',
                hoverongaps: false
            }};
            
            Plotly.newPlot('multi-video-plot', [trace], {{
                xaxis: {{title: 'Video IDs (Categorical)'}},
                yaxis: {{title: 'Video IDs (Categorical)'}},
                height: 400
            }}, {{responsive: true}});
        }}
    </script>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Meaningful similarity dashboard created: {output_path}")
        return output_path
    
    def _format_db_name(self, db_name: str) -> str:
        """Format database name for display"""
        if '5s' in db_name:
            return "5s Fragments"
        elif '20s' in db_name:
            return "20s Fragments"
        else:
            return "Full Videos"

def main():
    parser = argparse.ArgumentParser(description="Generate meaningful similarity dashboard")
    parser.add_argument("--db_paths", nargs='+', required=True, help="Database paths")
    parser.add_argument("--output", default="meaningful_similarity_dashboard.html", help="Output file")
    
    args = parser.parse_args()
    
    print("🔧 Meaningful EEG Similarity Analysis")
    print("=" * 40)
    
    analyzer = MeaningfulSimilarityAnalyzer(args.db_paths)
    output_path = analyzer.create_meaningful_dashboard(args.output)
    
    print(f"\n🌐 Open dashboard: file://{Path(output_path).absolute()}")

if __name__ == "__main__":
    main()
