#!/usr/bin/env python3
"""
Professional EEG Dashboard Generator
Creates a comprehensive HTML dashboard with search and visualization
"""

import numpy as np
import sqlite3
import pickle
import json
from pathlib import Path
from typing import Dict, List
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfessionalDashboardGenerator:
    def __init__(self, db_paths: List[str]):
        self.db_paths = [Path(path) for path in db_paths]
        self.databases = {}
        self.video_similarities = {}
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all databases and prepare similarity data"""
        for db_path in self.db_paths:
            if not db_path.exists():
                continue
                
            db_name = db_path.stem
            logger.info(f"📂 Loading: {db_name}")
            
            embeddings = []
            metadata = []
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(video_embeddings)")
                columns = [col[1] for col in cursor.fetchall()]
                is_fragment_db = 'fragment_id' in columns
                
                if is_fragment_db:
                    query = '''SELECT fragment_id, video_id, fragment_index, start_time_seconds, 
                               end_time_seconds, embedding_data, eeg_metrics, fragment_duration_seconds
                               FROM video_embeddings ORDER BY video_id, fragment_index'''
                else:
                    query = '''SELECT video_id, video_id, 0, 0, duration_seconds, 
                               embedding_data, eeg_metrics, duration_seconds
                               FROM video_embeddings ORDER BY video_id'''
                
                cursor.execute(query)
                for row in cursor.fetchall():
                    fragment_id, video_id, fragment_idx, start_time, end_time, embedding_data, eeg_metrics, duration = row
                    
                    embedding = pickle.loads(embedding_data)
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    embeddings.append(embedding)
                    eeg_dict = json.loads(eeg_metrics) if eeg_metrics else {}
                    attention = eeg_dict.get('attention', {})
                    
                    metadata.append({
                        'fragment_id': fragment_id,
                        'video_id': video_id,
                        'fragment_index': fragment_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'focus_level': attention.get('focus_level', 0.5),
                        'engagement': attention.get('engagement', 0.5),
                        'video_short': video_id.replace('lsl_stream_20250809_', '')
                    })
            
            self.databases[db_name] = {
                'embeddings': np.array(embeddings),
                'metadata': metadata,
                'fragment_duration': duration if embeddings else 0
            }
            
            # Calculate video similarities
            video_ids = sorted(list(set(meta['video_id'] for meta in metadata)))
            video_embeddings = {}
            
            for video_id in video_ids:
                video_indices = [i for i, meta in enumerate(metadata) if meta['video_id'] == video_id]
                if video_indices:
                    video_embs = np.array(embeddings)[video_indices]
                    video_embeddings[video_id] = np.mean(video_embs, axis=0)
            
            video_embedding_matrix = np.array([video_embeddings[vid] for vid in video_ids])
            similarity_matrix = cosine_similarity(video_embedding_matrix)
            
            self.video_similarities[db_name] = {
                'video_ids': video_ids,
                'video_short_names': [vid.replace('lsl_stream_20250809_', '') for vid in video_ids],
                'similarity_matrix': similarity_matrix.tolist(),
                'fragment_count': len(embeddings)
            }
            
            logger.info(f"✅ Processed {db_name}: {len(embeddings)} embeddings")
    
    def create_dashboard(self, output_path: str = "professional_eeg_dashboard.html"):
        """Create the professional dashboard"""
        logger.info("🎨 Creating professional dashboard...")
        
        # Prepare data
        stats = {
            'total_databases': len(self.databases),
            'total_fragments': sum(len(db['metadata']) for db in self.databases.values()),
            'unique_videos': len(set().union(*[set(meta['video_id'] for meta in db['metadata']) 
                                             for db in self.databases.values()])),
            'embedding_dimension': 768
        }
        
        # Get all video IDs
        all_video_ids = sorted(list(set().union(*[set(meta['video_id'] for meta in db['metadata']) 
                                                for db in self.databases.values()])))
        
        # Create HTML
        html_content = self._create_html_template(stats, all_video_ids)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Professional dashboard created: {output_path}")
        return output_path
    
    def _create_html_template(self, stats: Dict, video_ids: List[str]) -> str:
        """Create the HTML template"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional EEG Video Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .main-grid {{ display: grid; grid-template-columns: 300px 1fr; gap: 30px; }}
        .sidebar {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; }}
        .content {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; }}
        .control-group {{ margin-bottom: 20px; }}
        .control-group label {{ display: block; margin-bottom: 8px; font-weight: 600; }}
        .control-group select, .control-group input {{ width: 100%; padding: 10px; border: 2px solid #e0e0e0; border-radius: 8px; }}
        .search-btn {{ width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 10px; cursor: pointer; font-weight: 600; }}
        .video-list {{ background: #f8f9fa; border-radius: 10px; padding: 15px; margin-top: 20px; max-height: 300px; overflow-y: auto; }}
        .video-item {{ background: white; padding: 10px; margin: 5px 0; border-radius: 5px; cursor: pointer; border-left: 3px solid #667eea; }}
        .video-item:hover {{ background: #f0f8ff; }}
        .video-item.selected {{ background: #e3f2fd; border-left-color: #f5576c; }}
        .results {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; margin-top: 30px; display: none; }}
        .result-item {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #4caf50; }}
        .similarity-score {{ font-size: 1.2em; font-weight: bold; color: #4caf50; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> Professional EEG Video Analysis</h1>
            <p>Advanced Neural Pattern Search & Similarity Analysis</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">{stats['total_fragments']}</div><div>Total Fragments</div></div>
            <div class="stat-card"><div class="stat-value">{stats['unique_videos']}</div><div>Videos</div></div>
            <div class="stat-card"><div class="stat-value">{stats['total_databases']}</div><div>Databases</div></div>
            <div class="stat-card"><div class="stat-value">{stats['embedding_dimension']}</div><div>Dimensions</div></div>
        </div>
        
        <div class="main-grid">
            <div class="sidebar">
                <h3><i class="fas fa-search"></i> Search Controls</h3>
                
                <div class="control-group">
                    <label>Database:</label>
                    <select id="database-select" onchange="updateDatabase()">
                        {chr(10).join(f'<option value="{name}">{self._format_option(name, self.databases[name])}</option>' for name in self.databases.keys())}
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Query Video:</label>
                    <select id="video-select">
                        <option value="">Select video...</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Results:</label>
                    <input type="number" id="top-k" value="5" min="1" max="10">
                </div>
                
                <button class="search-btn" onclick="performSearch()">
                    <i class="fas fa-search"></i> Find Similar Videos
                </button>
                
                <div class="video-list">
                    <h4>📹 Video IDs</h4>
                    {chr(10).join(f'<div class="video-item" onclick="selectVideo(\'{vid}\')" data-video="{vid}">{i+1}. {vid.replace("lsl_stream_20250809_", "")}</div>' for i, vid in enumerate(video_ids))}
                </div>
            </div>
            
            <div class="content">
                <h3><i class="fas fa-chart-line"></i> Similarity Matrix</h3>
                <div id="heatmap-plot" style="height: 500px;"></div>
                
                <h3 style="margin-top: 30px;"><i class="fas fa-chart-bar"></i> Database Comparison</h3>
                <div id="comparison-plot" style="height: 400px;"></div>
            </div>
        </div>
        
        <div id="search-results" class="results">
            <h3><i class="fas fa-search-plus"></i> Search Results</h3>
            <div id="results-content"></div>
        </div>
    </div>

    <script>
        const databases = {json.dumps({name: {'similarity_matrix': sim['similarity_matrix'], 'video_ids': sim['video_ids'], 'video_short_names': sim['video_short_names'], 'fragment_count': sim['fragment_count']} for name, sim in self.video_similarities.items()})};
        const dbInfo = {json.dumps({name: {'duration': db['fragment_duration'], 'fragments': len(db['metadata'])} for name, db in self.databases.items()})};
        
        let currentDb = Object.keys(databases)[0];
        let selectedVideo = null;
        
        function updateDatabase() {{
            currentDb = document.getElementById('database-select').value;
            updateVideoSelector();
            createHeatmap();
            createComparison();
        }}
        
        function updateVideoSelector() {{
            const select = document.getElementById('video-select');
            const videoIds = databases[currentDb].video_ids;
            select.innerHTML = '<option value="">Select video...</option>';
            videoIds.forEach(id => {{
                const option = document.createElement('option');
                option.value = id;
                option.textContent = id.replace('lsl_stream_20250809_', '');
                select.appendChild(option);
            }});
        }}
        
        function selectVideo(videoId) {{
            selectedVideo = videoId;
            document.querySelectorAll('.video-item').forEach(item => item.classList.remove('selected'));
            document.querySelector(`[data-video="${{videoId}}"]`).classList.add('selected');
            document.getElementById('video-select').value = videoId;
        }}
        
        function createHeatmap() {{
            const data = databases[currentDb];
            const trace = {{
                z: data.similarity_matrix,
                x: data.video_short_names,
                y: data.video_short_names,
                type: 'heatmap',
                colorscale: 'Viridis',
                hovertemplate: 'Video Y: %{{y}}<br>Video X: %{{x}}<br>Similarity: %{{z:.4f}}<extra></extra>'
            }};
            
            const layout = {{
                title: `Video Similarity - ${{currentDb.replace('video_embeddings_', '').replace('s', '')}}s`,
                xaxis: {{title: 'Video IDs', tickangle: 45}},
                yaxis: {{title: 'Video IDs'}},
                height: 500
            }};
            
            Plotly.newPlot('heatmap-plot', [trace], layout, {{responsive: true}});
        }}
        
        function createComparison() {{
            const dbNames = Object.keys(databases);
            const trace = {{
                x: dbNames.map(name => name.replace('video_embeddings_', '').replace('s', '') + 's'),
                y: dbNames.map(name => databases[name].fragment_count),
                type: 'bar',
                marker: {{color: '#667eea'}}
            }};
            
            const layout = {{
                title: 'Fragment Count by Database',
                xaxis: {{title: 'Database Type'}},
                yaxis: {{title: 'Fragment Count'}},
                height: 400
            }};
            
            Plotly.newPlot('comparison-plot', [trace], layout, {{responsive: true}});
        }}
        
        function performSearch() {{
            const videoId = document.getElementById('video-select').value;
            const topK = document.getElementById('top-k').value;
            
            if (!videoId) {{
                alert('Please select a video first!');
                return;
            }}
            
            const data = databases[currentDb];
            const queryIdx = data.video_ids.indexOf(videoId);
            if (queryIdx === -1) return;
            
            const similarities = data.similarity_matrix[queryIdx];
            const results = [];
            
            similarities.forEach((sim, idx) => {{
                if (idx !== queryIdx) {{
                    results.push({{
                        video_id: data.video_ids[idx],
                        video_short: data.video_short_names[idx],
                        similarity: sim
                    }});
                }}
            }});
            
            results.sort((a, b) => b.similarity - a.similarity);
            
            const resultsHtml = results.slice(0, topK).map((result, i) => `
                <div class="result-item">
                    <div class="similarity-score">${{i+1}}. Similarity: ${{result.similarity.toFixed(4)}}</div>
                    <div><strong>Video:</strong> ${{result.video_short}}</div>
                    <div><strong>Full ID:</strong> ${{result.video_id}}</div>
                </div>
            `).join('');
            
            document.getElementById('results-content').innerHTML = resultsHtml;
            document.getElementById('search-results').style.display = 'block';
            document.getElementById('search-results').scrollIntoView({{behavior: 'smooth'}});
        }}
        
        document.addEventListener('DOMContentLoaded', function() {{
            updateVideoSelector();
            createHeatmap();
            createComparison();
        }});
    </script>
</body>
</html>"""
    
    def _format_option(self, name: str, db_data: Dict) -> str:
        """Format database option text"""
        duration = db_data['fragment_duration']
        count = len(db_data['metadata'])
        
        if duration == 0 or duration > 500:
            return f"Full Video ({count} videos)"
        else:
            return f"{duration:.0f}s Fragments ({count} total)"

def main():
    parser = argparse.ArgumentParser(description="Generate professional EEG dashboard")
    parser.add_argument("--db_paths", nargs='+', required=True, help="Database paths")
    parser.add_argument("--output", default="professional_eeg_dashboard.html", help="Output file")
    
    args = parser.parse_args()
    
    print("🎨 Professional EEG Dashboard Generator")
    print("=" * 50)
    
    generator = ProfessionalDashboardGenerator(args.db_paths)
    output_path = generator.create_dashboard(args.output)
    
    print(f"\n🌐 Open in browser: file://{Path(output_path).absolute()}")

if __name__ == "__main__":
    main()
