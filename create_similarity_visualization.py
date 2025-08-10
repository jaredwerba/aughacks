#!/usr/bin/env python3
"""
Create Interactive HTML Visualization for Video Similarity
Generates a web-based dashboard to explore video similarity patterns
"""

import numpy as np
import sqlite3
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

class SimilarityVisualizationGenerator:
    """Generate interactive HTML visualizations for video similarity"""
    
    def __init__(self, db_paths: List[str]):
        self.db_paths = [Path(path) for path in db_paths]
        for db_path in self.db_paths:
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.databases = {}  # Store data for each database
        self._load_all_databases()
    
    def _load_all_databases(self):
        """Load embeddings and metadata from all databases"""
        for db_path in self.db_paths:
            db_name = db_path.stem  # e.g., 'video_embeddings_5s'
            print(f"📂 Loading embeddings from: {db_path}")
            
            embeddings_list = []
            ids_list = []
            metadata_cache = {}
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check if this is a fragment database or full video database
                cursor.execute("PRAGMA table_info(video_embeddings)")
                columns = [col[1] for col in cursor.fetchall()]
                is_fragment_db = 'fragment_id' in columns
                
                if is_fragment_db:
                    query = '''
                        SELECT fragment_id, video_id, fragment_index, start_time_seconds, 
                               end_time_seconds, embedding_data, eeg_metrics, video_title,
                               fragment_duration_seconds, total_video_duration_seconds
                        FROM video_embeddings 
                        ORDER BY video_id, fragment_index
                    '''
                else:
                    query = '''
                        SELECT video_id, video_id as fragment_id, 0 as fragment_index, 
                               0 as start_time_seconds, duration_seconds as end_time_seconds,
                               embedding_data, eeg_metrics, video_title,
                               duration_seconds as fragment_duration_seconds, 
                               duration_seconds as total_video_duration_seconds
                        FROM video_embeddings 
                        ORDER BY video_id
                    '''
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                for row in results:
                    (fragment_id, video_id, fragment_index, start_time, end_time, 
                     embedding_data, eeg_metrics, video_title, fragment_duration, total_duration) = row
                    
                    # Load embedding
                    embedding = pickle.loads(embedding_data)
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    embeddings_list.append(embedding)
                    ids_list.append(fragment_id)
                    
                    # Store metadata
                    metadata_cache[fragment_id] = {
                        'video_id': video_id,
                        'fragment_index': fragment_index,
                        'start_time': start_time,
                        'end_time': end_time,
                        'video_title': video_title,
                        'fragment_duration': fragment_duration,
                        'total_duration': total_duration,
                        'eeg_metrics': json.loads(eeg_metrics) if eeg_metrics else {},
                        'database': db_name
                    }
                
                # Store data for this database
                self.databases[db_name] = {
                    'embeddings_matrix': np.array(embeddings_list),
                    'embedding_ids': ids_list,
                    'metadata_cache': metadata_cache,
                    'fragment_duration': fragment_duration if embeddings_list else 0
                }
                
                print(f"✅ Loaded {len(embeddings_list)} embeddings from {db_name}")
                print(f"📊 Embedding shape: {np.array(embeddings_list).shape}")
    
    def generate_similarity_data(self):
        """Generate similarity matrix and dimensional reduction data for all databases"""
        print("🧮 Calculating similarity matrices for all databases...")
        
        all_data = {}
        
        for db_name, db_data in self.databases.items():
            print(f"📊 Processing {db_name}...")
            
            embeddings_matrix = db_data['embeddings_matrix']
            embedding_ids = db_data['embedding_ids']
            metadata_cache = db_data['metadata_cache']
            
            # Get all unique video IDs and sort them for consistent ordering
            video_ids = sorted(list(set(meta['video_id'] for meta in metadata_cache.values())))
            
            # Calculate video-level similarities (average fragments per video)
            video_embeddings = {}
            video_metadata = {}
            
            for video_id in video_ids:
                video_fragments = [embeddings_matrix[i] for i, fid in enumerate(embedding_ids) 
                                 if metadata_cache[fid]['video_id'] == video_id]
                if video_fragments:
                    video_embeddings[video_id] = np.mean(video_fragments, axis=0)
                    # Get representative metadata
                    rep_fragment = next(fid for fid in embedding_ids if metadata_cache[fid]['video_id'] == video_id)
                    video_metadata[video_id] = metadata_cache[rep_fragment]
            
            # Create video similarity matrix with proper ordering
            video_embedding_list = [video_embeddings[vid] for vid in video_ids]
            video_similarity_matrix = cosine_similarity(video_embedding_list)
            
            # Create short video names for display
            video_short_names = [vid.replace('lsl_stream_20250809_', '') for vid in video_ids]
            
            print(f"🎯 Performing dimensional reduction for {db_name}...")
            
            # PCA for 2D visualization
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_matrix)
            
            # t-SNE for 2D visualization (sample if too many points)
            sample_size = min(300, len(embeddings_matrix))
            sample_indices = np.random.choice(len(embeddings_matrix), sample_size, replace=False)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
            embeddings_tsne = tsne.fit_transform(embeddings_matrix[sample_indices])
            
            all_data[db_name] = {
                'video_similarity_matrix': video_similarity_matrix.tolist(),
                'video_ids': video_ids,
                'video_short_names': video_short_names,
                'video_metadata': video_metadata,
                'embeddings_2d_pca': embeddings_2d.tolist(),
                'embeddings_2d_tsne': embeddings_tsne.tolist(),
                'tsne_sample_indices': sample_indices.tolist(),
                'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
                'fragment_duration': db_data['fragment_duration'],
                'total_fragments': len(embedding_ids),
                'embedding_ids': embedding_ids,
                'metadata_cache': metadata_cache
            }
        
        return all_data
    
    def create_html_visualization(self, output_path: str = "video_similarity_dashboard.html"):
        """Create interactive HTML visualization"""
        print("🎨 Generating interactive HTML visualization...")
        
        # Generate similarity data for all databases
        all_sim_data = self.generate_similarity_data()
        
        # Get the first database for initial stats
        first_db_name = list(all_sim_data.keys())[0]
        first_db_data = all_sim_data[first_db_name]
        
        # Calculate total stats across all databases
        total_fragments = sum(data['total_fragments'] for data in all_sim_data.values())
        unique_videos = len(first_db_data['video_ids'])  # Same videos across all databases
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG Video Similarity Dashboard</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}
        
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .visualization-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .viz-container {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }}
        
        .viz-title {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #667eea;
            text-align: center;
        }}
        
        .controls {{
            margin-bottom: 30px;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 10px;
            border-left: 5px solid #2196f3;
        }}
        
        .control-group {{
            margin-bottom: 15px;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #1976d2;
        }}
        
        .control-group select, .control-group input {{
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }}
        
        .similarity-results {{
            margin-top: 20px;
            padding: 20px;
            background: #f1f8e9;
            border-radius: 10px;
            border-left: 5px solid #4caf50;
        }}
        
        .result-item {{
            padding: 10px;
            margin: 10px 0;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #4caf50;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .similarity-score {{
            font-weight: bold;
            color: #2e7d32;
            font-size: 1.1em;
        }}
        
        #plotly-div {{
            width: 100%;
            height: 500px;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 EEG Video Similarity Dashboard</h1>
            <p>Interactive exploration of video embedding similarities</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-embeddings">{len(fragments_data)}</div>
                <div class="stat-label">Total Fragments</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="unique-videos">{len(set(f['video_id'] for f in fragments_data))}</div>
                <div class="stat-label">Unique Videos</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="embedding-dim">768</div>
                <div class="stat-label">Embedding Dimensions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-duration">{np.mean([f['duration'] for f in fragments_data]):.1f}s</div>
                <div class="stat-label">Avg Fragment Duration</div>
            </div>
        </div>
        
        <div class="controls">
            <h3>🎯 Similarity Search Controls</h3>
            <div class="control-group">
                <label for="video-select">Select Video/Fragment:</label>
                <select id="video-select">
                    <option value="">Choose a video or fragment...</option>
                </select>
            </div>
            <div class="control-group">
                <label for="search-type">Search Type:</label>
                <select id="search-type">
                    <option value="fragment">Fragment Similarity</option>
                    <option value="video">Video Similarity</option>
                </select>
            </div>
            <div class="control-group">
                <label for="top-k">Number of Results:</label>
                <input type="number" id="top-k" value="5" min="1" max="20">
            </div>
            <button onclick="performSimilaritySearch()" style="background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px;">
                🔍 Search Similar Videos
            </button>
        </div>
        
        <div class="visualization-grid">
            <div class="viz-container">
                <div class="viz-title">📊 PCA Visualization (2D)</div>
                <div id="pca-plot"></div>
            </div>
            <div class="viz-container">
                <div class="viz-title">🎯 t-SNE Clustering (2D)</div>
                <div id="tsne-plot"></div>
            </div>
            <div class="viz-container full-width">
                <div class="viz-title">🔥 Video Similarity Heatmap</div>
                <div id="heatmap-plot"></div>
            </div>
        </div>
        
        <div id="similarity-results" class="similarity-results" style="display: none;">
            <h3>🎯 Similarity Search Results</h3>
            <div id="results-container"></div>
        </div>
    </div>

    <script>
        // Data from Python
        const fragmentsData = {json.dumps(fragments_data, indent=2)};
        const videoSimilarityData = {json.dumps(video_sim_data, indent=2)};
        
        // Populate video selector
        function populateVideoSelector() {{
            const select = document.getElementById('video-select');
            const videos = [...new Set(fragmentsData.map(f => f.video_id))];
            
            // Add video options
            videos.forEach(videoId => {{
                const option = document.createElement('option');
                option.value = videoId;
                option.textContent = `📹 ${{videoId.replace('lsl_stream_20250809_', '')}} (Video)`;
                select.appendChild(option);
            }});
            
            // Add fragment options (first 50 for performance)
            fragmentsData.slice(0, 50).forEach(fragment => {{
                const option = document.createElement('option');
                option.value = fragment.id;
                option.textContent = `🎬 ${{fragment.video_short}}_f${{fragment.fragment_index.toString().padStart(3, '0')}} (${{fragment.start_time}}s-${{fragment.end_time}}s)`;
                select.appendChild(option);
            }});
        }}
        
        // Create PCA visualization
        function createPCAPlot() {{
            const trace = {{
                x: fragmentsData.map(f => f.pca_x),
                y: fragmentsData.map(f => f.pca_y),
                mode: 'markers',
                type: 'scatter',
                text: fragmentsData.map(f => `${{f.video_short}}_f${{f.fragment_index}}<br>Focus: ${{f.focus_level.toFixed(3)}}<br>Time: ${{f.start_time}}s-${{f.end_time}}s`),
                marker: {{
                    size: 8,
                    color: fragmentsData.map(f => f.focus_level),
                    colorscale: 'Viridis',
                    colorbar: {{title: 'Focus Level'}},
                    line: {{width: 1, color: 'white'}}
                }},
                hovertemplate: '%{{text}}<extra></extra>'
            }};
            
            const layout = {{
                title: 'PCA Projection of EEG Embeddings',
                xaxis: {{title: 'PC1'}},
                yaxis: {{title: 'PC2'}},
                hovermode: 'closest',
                height: 400
            }};
            
            Plotly.newPlot('pca-plot', [trace], layout, {{responsive: true}});
        }}
        
        // Create t-SNE visualization
        function createTSNEPlot() {{
            const tsneFragments = fragmentsData.filter(f => f.tsne_x !== undefined);
            
            const trace = {{
                x: tsneFragments.map(f => f.tsne_x),
                y: tsneFragments.map(f => f.tsne_y),
                mode: 'markers',
                type: 'scatter',
                text: tsneFragments.map(f => `${{f.video_short}}_f${{f.fragment_index}}<br>Focus: ${{f.focus_level.toFixed(3)}}<br>Time: ${{f.start_time}}s-${{f.end_time}}s`),
                marker: {{
                    size: 10,
                    color: tsneFragments.map(f => f.engagement),
                    colorscale: 'Plasma',
                    colorbar: {{title: 'Engagement'}},
                    line: {{width: 1, color: 'white'}}
                }},
                hovertemplate: '%{{text}}<extra></extra>'
            }};
            
            const layout = {{
                title: 't-SNE Clustering of EEG Embeddings',
                xaxis: {{title: 't-SNE 1'}},
                yaxis: {{title: 't-SNE 2'}},
                hovermode: 'closest',
                height: 400
            }};
            
            Plotly.newPlot('tsne-plot', [trace], layout, {{responsive: true}});
        }}
        
        // Create video similarity heatmap
        function createSimilarityHeatmap() {{
            const videos = [...new Set(fragmentsData.map(f => f.video_id))];
            const shortNames = videos.map(v => v.replace('lsl_stream_20250809_', ''));
            
            // Create similarity matrix from video similarity data
            const matrix = Array(videos.length).fill().map(() => Array(videos.length).fill(0));
            
            videoSimilarityData.forEach(d => {{
                const sourceIdx = shortNames.indexOf(d.source);
                const targetIdx = shortNames.indexOf(d.target);
                if (sourceIdx >= 0 && targetIdx >= 0) {{
                    matrix[sourceIdx][targetIdx] = d.similarity;
                }}
            }});
            
            // Set diagonal to 1 (self-similarity)
            for (let i = 0; i < matrix.length; i++) {{
                matrix[i][i] = 1.0;
            }}
            
            const trace = {{
                z: matrix,
                x: shortNames,
                y: shortNames,
                type: 'heatmap',
                colorscale: 'Viridis',
                hoverongaps: false,
                hovertemplate: 'Video 1: %{{y}}<br>Video 2: %{{x}}<br>Similarity: %{{z:.4f}}<extra></extra>'
            }};
            
            const layout = {{
                title: 'Video-to-Video Similarity Matrix',
                xaxis: {{title: 'Videos'}},
                yaxis: {{title: 'Videos'}},
                height: 500
            }};
            
            Plotly.newPlot('heatmap-plot', [trace], layout, {{responsive: true}});
        }}
        
        // Perform similarity search (mock implementation for demo)
        function performSimilaritySearch() {{
            const selectedId = document.getElementById('video-select').value;
            const searchType = document.getElementById('search-type').value;
            const topK = document.getElementById('top-k').value;
            
            if (!selectedId) {{
                alert('Please select a video or fragment first!');
                return;
            }}
            
            // Mock similarity results for demonstration
            const resultsContainer = document.getElementById('results-container');
            const resultsDiv = document.getElementById('similarity-results');
            
            resultsContainer.innerHTML = `
                <div class="result-item">
                    <div class="similarity-score">🎯 Query: ${{selectedId}}</div>
                    <div>Search Type: ${{searchType}}</div>
                    <div>Top ${{topK}} results requested</div>
                </div>
                <div class="result-item">
                    <div class="similarity-score">Similarity: 0.9876</div>
                    <div>📹 Video: lsl_stream_20250809_155123</div>
                    <div>⏱️ Time: 125.0s - 130.0s</div>
                    <div>🧠 Focus: 0.789</div>
                </div>
                <div class="result-item">
                    <div class="similarity-score">Similarity: 0.9654</div>
                    <div>📹 Video: lsl_stream_20250809_160350</div>
                    <div>⏱️ Time: 45.0s - 50.0s</div>
                    <div>🧠 Focus: 0.623</div>
                </div>
                <p><em>💡 Note: This is a demo interface. For actual similarity search, use the Python command-line tool.</em></p>
            `;
            
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({{ behavior: 'smooth' }});
        }}
        
        // Initialize visualizations
        document.addEventListener('DOMContentLoaded', function() {{
            populateVideoSelector();
            createPCAPlot();
            createTSNEPlot();
            createSimilarityHeatmap();
        }});
    </script>
</body>
</html>
        """
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Interactive HTML visualization created: {output_path}")
        return output_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate interactive HTML visualization")
    parser.add_argument("--db_paths", nargs='+', required=True, 
                       help="Paths to the embeddings databases (e.g., video_embeddings_5s.db video_embeddings_20s.db)")
    parser.add_argument("--output", default="video_similarity_dashboard.html", 
                       help="Output HTML file path")
    
    args = parser.parse_args()
    
    print("🎨 Creating Interactive Video Similarity Visualization")
    print("=" * 60)
    
    # Generate visualization
    generator = SimilarityVisualizationGenerator(args.db_paths)
    output_path = generator.create_html_visualization(args.output)
    
    print(f"\n🌐 Open the following file in your browser:")
    print(f"   file://{Path(output_path).absolute()}")

if __name__ == "__main__":
    main()
