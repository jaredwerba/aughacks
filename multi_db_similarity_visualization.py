#!/usr/bin/env python3
"""
Create Interactive HTML Visualization for Multiple Video Similarity Databases
Generates a web-based dashboard to compare video similarity patterns across different fragment durations
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

class MultiDatabaseVisualizationGenerator:
    """Generate interactive HTML visualizations for multiple video similarity databases"""
    
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
    
    def generate_video_similarity_matrices(self):
        """Generate video-to-video similarity matrices for all databases"""
        print("🧮 Calculating video similarity matrices for all databases...")
        
        all_matrices = {}
        
        for db_name, db_data in self.databases.items():
            print(f"📊 Processing {db_name}...")
            
            embeddings_matrix = db_data['embeddings_matrix']
            embedding_ids = db_data['embedding_ids']
            metadata_cache = db_data['metadata_cache']
            
            # Get all unique video IDs and sort them for consistent ordering
            video_ids = sorted(list(set(meta['video_id'] for meta in metadata_cache.values())))
            
            # Calculate video-level embeddings (average fragments per video)
            video_embeddings = {}
            
            for video_id in video_ids:
                video_fragments = [embeddings_matrix[i] for i, fid in enumerate(embedding_ids) 
                                 if metadata_cache[fid]['video_id'] == video_id]
                if video_fragments:
                    video_embeddings[video_id] = np.mean(video_fragments, axis=0)
            
            # Create video similarity matrix with proper ordering
            video_embedding_list = [video_embeddings[vid] for vid in video_ids]
            video_similarity_matrix = cosine_similarity(video_embedding_list)
            
            # Create short video names for display (showing the timestamp part)
            video_short_names = [vid.replace('lsl_stream_20250809_', '') for vid in video_ids]
            
            all_matrices[db_name] = {
                'similarity_matrix': video_similarity_matrix.tolist(),
                'video_ids': video_ids,
                'video_short_names': video_short_names,
                'fragment_duration': db_data['fragment_duration'],
                'total_fragments': len(embedding_ids)
            }
        
        return all_matrices
    
    def create_html_visualization(self, output_path: str = "video_similarity_dashboard.html"):
        """Create interactive HTML visualization"""
        print("🎨 Generating interactive HTML visualization...")
        
        # Generate video similarity matrices
        all_matrices = self.generate_video_similarity_matrices()
        
        # Get database names and info
        db_names = list(all_matrices.keys())
        db_info = {name: {'duration': data['fragment_duration'], 'fragments': data['total_fragments']} 
                  for name, data in all_matrices.items()}
        
        # Get video IDs (consistent across all databases)
        video_ids = all_matrices[db_names[0]]['video_ids']
        video_short_names = all_matrices[db_names[0]]['video_short_names']
        
        # Calculate total stats
        total_fragments = sum(data['total_fragments'] for data in all_matrices.values())
        unique_videos = len(video_ids)
        
        # Helper function to format database options
        def format_database_option(name, data):
            if name == "video_embeddings":
                return f'<option value="{name}">Full Video Embeddings ({data["fragments"]} videos - Complete Sessions)</option>'
            elif "5s" in name:
                return f'<option value="{name}">5s Fragments ({data["fragments"]} total - High Resolution)</option>'
            elif "20s" in name:
                return f'<option value="{name}">20s Fragments ({data["fragments"]} total - Medium Resolution)</option>'
            else:
                duration = name.replace("video_embeddings_", "").replace("s", "")
                return f'<option value="{name}">{duration}s Fragments ({data["fragments"]} total)</option>'
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG Video Similarity Multi-Database Dashboard</title>
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
            max-width: 1600px;
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
        
        .database-selector {{
            margin-bottom: 30px;
            padding: 20px;
            background: #e8f5e8;
            border-radius: 10px;
            border-left: 5px solid #4caf50;
        }}
        
        .database-selector label {{
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #2e7d32;
        }}
        
        .database-selector select {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }}
        
        .heatmap-container {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            margin-bottom: 30px;
        }}
        
        .viz-title {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 20px;
            color: #667eea;
            text-align: center;
        }}
        
        .video-ids-display {{
            background: #fff3e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 5px solid #ff9800;
        }}
        
        .video-id-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        
        .video-id-item {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #ff9800;
            font-family: monospace;
            font-size: 0.9em;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        @media (max-width: 1200px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        #plotly-div {{
            width: 100%;
            height: 600px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 EEG Video Similarity Multi-Database Dashboard</h1>
            <p>Compare video embedding similarities across different fragment durations</p>
            
            <div style="background: #e1f5fe; padding: 20px; border-radius: 10px; margin-top: 20px; text-align: left;">
                <h3 style="color: #0277bd; margin-top: 0;">📖 What Each Dashboard Represents:</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50;">
                        <strong>🔸 5s Fragments (996 total)</strong><br>
                        <em>High temporal resolution</em><br>
                        Captures rapid neural state changes and micro-patterns in EEG signals
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800;">
                        <strong>🔸 20s Fragments (243 total)</strong><br>
                        <em>Medium temporal resolution</em><br>
                        Captures sustained neural patterns and cognitive state transitions
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #9c27b0;">
                        <strong>🔸 Full Video (10 total)</strong><br>
                        <em>Complete session analysis</em><br>
                        Represents overall neural signature and cognitive profile per video
                    </div>
                </div>
                <p style="margin-bottom: 0; margin-top: 15px; font-style: italic; color: #0277bd;">
                    💡 <strong>Key Insight:</strong> Higher similarity scores between videos in different temporal scales indicate robust neural patterns that persist across time.
                </p>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_fragments}</div>
                <div class="stat-label">Total Fragments</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{unique_videos}</div>
                <div class="stat-label">Unique Videos</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(db_names)}</div>
                <div class="stat-label">Databases</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">768</div>
                <div class="stat-label">Embedding Dimensions</div>
            </div>
        </div>
        
        <div class="video-ids-display">
            <h3>📹 Video IDs in Dataset</h3>
            <p>The following 10 videos are analyzed across all fragment durations:</p>
            <div class="video-id-grid">
                {chr(10).join(f'<div class="video-id-item"><strong>{i+1}.</strong> {vid}</div>' for i, vid in enumerate(video_ids))}
            </div>
        </div>
        
        <div class="database-selector">
            <h3>🗄️ Database Selection</h3>
            <label for="database-select">Select Fragment Duration Database:</label>
            <select id="database-select" onchange="updateVisualization()">
                {chr(10).join(format_database_option(name, data) for name, data in db_info.items())}
            </select>
            
            <div style="margin-top: 15px; padding: 15px; background: #fff3e0; border-radius: 8px; border-left: 4px solid #ff9800;">
                <strong>🎯 Current View:</strong> <span id="current-description">Select a database to see its description</span>
            </div>
        </div>
        
        <div class="comparison-grid">
            <div class="heatmap-container">
                <div class="viz-title">🔥 Video Similarity Heatmap</div>
                <div id="heatmap-plot"></div>
            </div>
            <div class="heatmap-container">
                <div class="viz-title">📊 Similarity Statistics</div>
                <div id="stats-plot"></div>
            </div>
        </div>
        
        <div class="heatmap-container">
            <div class="viz-title">📈 Cross-Database Comparison</div>
            <div id="comparison-plot"></div>
        </div>
    </div>

    <script>
        // Data from Python
        const allMatricesData = {json.dumps(all_matrices, indent=2)};
        const videoIds = {json.dumps(video_ids)};
        const videoShortNames = {json.dumps(video_short_names)};
        const dbInfo = {json.dumps(db_info)};
        
        let currentDatabase = Object.keys(allMatricesData)[0];
        
        // Create video similarity heatmap
        function createSimilarityHeatmap(dbName) {{
            const data = allMatricesData[dbName];
            
            const trace = {{
                z: data.similarity_matrix,
                x: data.video_short_names,
                y: data.video_short_names,
                type: 'heatmap',
                colorscale: 'Viridis',
                hoverongaps: false,
                hovertemplate: 'Video Y: %{{y}}<br>Video X: %{{x}}<br>Similarity: %{{z:.4f}}<extra></extra>',
                colorbar: {{
                    title: 'Cosine Similarity',
                    titleside: 'right'
                }}
            }};
            
            const layout = {{
                title: `Video Similarity Matrix - ${{dbName.replace('video_embeddings_', '').replace('s', '')}}s Fragments`,
                xaxis: {{
                    title: 'Video IDs',
                    tickangle: 45,
                    tickfont: {{size: 10}}
                }},
                yaxis: {{
                    title: 'Video IDs',
                    tickfont: {{size: 10}}
                }},
                height: 500,
                margin: {{l: 100, r: 50, t: 80, b: 100}}
            }};
            
            Plotly.newPlot('heatmap-plot', [trace], layout, {{responsive: true}});
        }}
        
        // Create similarity statistics plot
        function createStatsPlot(dbName) {{
            const data = allMatricesData[dbName];
            const matrix = data.similarity_matrix;
            
            // Calculate statistics for each video
            const stats = data.video_short_names.map((name, i) => {{
                const row = matrix[i];
                const otherSimilarities = row.filter((_, j) => i !== j); // Exclude self-similarity
                return {{
                    video: name,
                    avg_similarity: otherSimilarities.reduce((a, b) => a + b, 0) / otherSimilarities.length,
                    max_similarity: Math.max(...otherSimilarities),
                    min_similarity: Math.min(...otherSimilarities)
                }};
            }});
            
            const trace1 = {{
                x: stats.map(s => s.video),
                y: stats.map(s => s.avg_similarity),
                type: 'bar',
                name: 'Average Similarity',
                marker: {{color: '#667eea'}}
            }};
            
            const trace2 = {{
                x: stats.map(s => s.video),
                y: stats.map(s => s.max_similarity),
                type: 'scatter',
                mode: 'markers',
                name: 'Max Similarity',
                marker: {{color: '#f5576c', size: 8}}
            }};
            
            const layout = {{
                title: `Similarity Statistics - ${{dbName.replace('video_embeddings_', '').replace('s', '')}}s Fragments`,
                xaxis: {{
                    title: 'Video IDs',
                    tickangle: 45
                }},
                yaxis: {{title: 'Similarity Score'}},
                height: 500,
                showlegend: true
            }};
            
            Plotly.newPlot('stats-plot', [trace1, trace2], layout, {{responsive: true}});
        }}
        
        // Create cross-database comparison
        function createComparisonPlot() {{
            const dbNames = Object.keys(allMatricesData);
            
            // Calculate average similarity for each database
            const comparisonData = dbNames.map(dbName => {{
                const matrix = allMatricesData[dbName].similarity_matrix;
                const n = matrix.length;
                let totalSim = 0;
                let count = 0;
                
                for (let i = 0; i < n; i++) {{
                    for (let j = 0; j < n; j++) {{
                        if (i !== j) {{
                            totalSim += matrix[i][j];
                            count++;
                        }}
                    }}
                }}
                
                return {{
                    database: dbName.replace('video_embeddings_', '').replace('s', '') + 's',
                    avg_similarity: totalSim / count,
                    fragment_count: allMatricesData[dbName].total_fragments,
                    duration: allMatricesData[dbName].fragment_duration
                }};
            }});
            
            const trace1 = {{
                x: comparisonData.map(d => d.database),
                y: comparisonData.map(d => d.avg_similarity),
                type: 'bar',
                name: 'Avg Similarity',
                marker: {{color: '#4caf50'}},
                yaxis: 'y'
            }};
            
            const trace2 = {{
                x: comparisonData.map(d => d.database),
                y: comparisonData.map(d => d.fragment_count),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Fragment Count',
                marker: {{color: '#ff9800', size: 8}},
                line: {{color: '#ff9800', width: 3}},
                yaxis: 'y2'
            }};
            
            const layout = {{
                title: 'Cross-Database Comparison',
                xaxis: {{title: 'Fragment Duration'}},
                yaxis: {{
                    title: 'Average Similarity',
                    side: 'left'
                }},
                yaxis2: {{
                    title: 'Fragment Count',
                    side: 'right',
                    overlaying: 'y'
                }},
                height: 500,
                showlegend: true
            }};
            
            Plotly.newPlot('comparison-plot', [trace1, trace2], layout, {{responsive: true}});
        }}
        
        // Update visualization based on selected database
        function updateVisualization() {{
            const selectedDb = document.getElementById('database-select').value;
            currentDatabase = selectedDb;
            
            createSimilarityHeatmap(selectedDb);
            createStatsPlot(selectedDb);
        }}
        
        // Initialize visualizations
        document.addEventListener('DOMContentLoaded', function() {{
            createSimilarityHeatmap(currentDatabase);
            createStatsPlot(currentDatabase);
            createComparisonPlot();
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
    parser = argparse.ArgumentParser(description="Generate interactive HTML visualization for multiple databases")
    parser.add_argument("--db_paths", nargs='+', required=True, 
                       help="Paths to the embeddings databases (e.g., video_embeddings_5s.db video_embeddings_20s.db)")
    parser.add_argument("--output", default="multi_db_similarity_dashboard.html", 
                       help="Output HTML file path")
    
    args = parser.parse_args()
    
    print("🎨 Creating Multi-Database Video Similarity Visualization")
    print("=" * 70)
    
    # Generate visualization
    generator = MultiDatabaseVisualizationGenerator(args.db_paths)
    output_path = generator.create_html_visualization(args.output)
    
    print(f"\n🌐 Open the following file in your browser:")
    print(f"   file://{Path(output_path).absolute()}")

if __name__ == "__main__":
    main()
