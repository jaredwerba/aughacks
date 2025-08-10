#!/usr/bin/env python3
"""
Attention Statistics Extractor
Extracts and analyzes attention model statistics from EEG video databases
"""

import numpy as np
import sqlite3
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import pandas as pd
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttentionStatsExtractor:
    """Extract and analyze attention model statistics from EEG databases"""
    
    def __init__(self, db_paths: List[str]):
        self.db_paths = [Path(path) for path in db_paths]
        self.attention_data = {}
        self.video_stats = {}
        self._extract_attention_data()
        self._calculate_video_statistics()
    
    def _extract_attention_data(self):
        """Extract attention data from all databases"""
        logger.info("📊 Extracting attention model statistics...")
        
        for db_path in self.db_paths:
            if not db_path.exists():
                continue
                
            db_name = db_path.stem
            logger.info(f"🔍 Processing: {db_name}")
            
            attention_records = []
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check schema
                cursor.execute("PRAGMA table_info(video_embeddings)")
                columns = [col[1] for col in cursor.fetchall()]
                is_fragment_db = 'fragment_id' in columns
                
                if is_fragment_db:
                    query = '''
                        SELECT fragment_id, video_id, fragment_index, start_time_seconds, 
                               end_time_seconds, eeg_metrics, fragment_duration_seconds
                        FROM video_embeddings ORDER BY video_id, fragment_index
                    '''
                else:
                    query = '''
                        SELECT video_id, video_id, 0, 0, 
                               duration_seconds, eeg_metrics, duration_seconds
                        FROM video_embeddings ORDER BY video_id
                    '''
                
                cursor.execute(query)
                for row in cursor.fetchall():
                    if is_fragment_db:
                        fragment_id, video_id, fragment_idx, start_time, end_time, eeg_metrics, duration = row
                    else:
                        fragment_id, video_id, fragment_idx, start_time, duration, eeg_metrics, _ = row
                        end_time = duration
                    
                    # Parse EEG metrics
                    eeg_dict = json.loads(eeg_metrics) if eeg_metrics else {}
                    attention = eeg_dict.get('attention', {})
                    
                    # Extract all attention metrics
                    attention_record = {
                        'database': db_name,
                        'fragment_id': fragment_id,
                        'video_id': video_id,
                        'video_short': video_id.replace('lsl_stream_20250809_', ''),
                        'fragment_index': fragment_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'focus_level': attention.get('focus_level', 0.0),
                        'engagement': attention.get('engagement', 0.0),
                        'attention_alpha': attention.get('alpha_power', 0.0),
                        'attention_beta': attention.get('beta_power', 0.0),
                        'attention_theta': attention.get('theta_power', 0.0),
                        'attention_gamma': attention.get('gamma_power', 0.0),
                        'meditation_level': attention.get('meditation', 0.0),
                        'mental_effort': attention.get('mental_effort', 0.0),
                        'cognitive_workload': attention.get('cognitive_workload', 0.0),
                        'alertness': attention.get('alertness', 0.0),
                        'drowsiness': attention.get('drowsiness', 0.0),
                        'stress_level': attention.get('stress_level', 0.0)
                    }
                    
                    attention_records.append(attention_record)
            
            self.attention_data[db_name] = attention_records
            logger.info(f"✅ Extracted {len(attention_records)} attention records from {db_name}")
    
    def _calculate_video_statistics(self):
        """Calculate comprehensive statistics per video"""
        logger.info("📈 Calculating video-level attention statistics...")
        
        for db_name, records in self.attention_data.items():
            df = pd.DataFrame(records)
            video_stats = {}
            
            # Group by video
            for video_id in df['video_id'].unique():
                video_data = df[df['video_id'] == video_id]
                
                # Calculate comprehensive statistics
                stats_dict = {
                    'video_id': video_id,
                    'video_short': video_id.replace('lsl_stream_20250809_', ''),
                    'database': db_name,
                    'total_fragments': len(video_data),
                    'total_duration': video_data['duration'].sum(),
                    'avg_fragment_duration': video_data['duration'].mean(),
                    
                    # Focus statistics
                    'focus_mean': video_data['focus_level'].mean(),
                    'focus_std': video_data['focus_level'].std(),
                    'focus_min': video_data['focus_level'].min(),
                    'focus_max': video_data['focus_level'].max(),
                    'focus_median': video_data['focus_level'].median(),
                    'focus_q25': video_data['focus_level'].quantile(0.25),
                    'focus_q75': video_data['focus_level'].quantile(0.75),
                    
                    # Engagement statistics
                    'engagement_mean': video_data['engagement'].mean(),
                    'engagement_std': video_data['engagement'].std(),
                    'engagement_min': video_data['engagement'].min(),
                    'engagement_max': video_data['engagement'].max(),
                    'engagement_median': video_data['engagement'].median(),
                    'engagement_q25': video_data['engagement'].quantile(0.25),
                    'engagement_q75': video_data['engagement'].quantile(0.75),
                    
                    # Brainwave power statistics
                    'alpha_mean': video_data['attention_alpha'].mean(),
                    'beta_mean': video_data['attention_beta'].mean(),
                    'theta_mean': video_data['attention_theta'].mean(),
                    'gamma_mean': video_data['attention_gamma'].mean(),
                    
                    # Additional cognitive metrics
                    'meditation_mean': video_data['meditation_level'].mean(),
                    'mental_effort_mean': video_data['mental_effort'].mean(),
                    'cognitive_workload_mean': video_data['cognitive_workload'].mean(),
                    'alertness_mean': video_data['alertness'].mean(),
                    'drowsiness_mean': video_data['drowsiness'].mean(),
                    'stress_mean': video_data['stress_level'].mean(),
                    
                    # Temporal analysis
                    'focus_trend': self._calculate_trend(video_data['focus_level'].values),
                    'engagement_trend': self._calculate_trend(video_data['engagement'].values),
                    'focus_stability': self._calculate_stability(video_data['focus_level'].values),
                    'engagement_stability': self._calculate_stability(video_data['engagement'].values),
                    
                    # Peak analysis
                    'focus_peaks': self._count_peaks(video_data['focus_level'].values),
                    'engagement_peaks': self._count_peaks(video_data['engagement'].values),
                    
                    # Correlation analysis
                    'focus_engagement_corr': video_data['focus_level'].corr(video_data['engagement']),
                    
                    # Performance classification
                    'performance_category': self._classify_performance(video_data),
                    'attention_profile': self._classify_attention_profile(video_data)
                }
                
                video_stats[video_id] = stats_dict
            
            self.video_stats[db_name] = video_stats
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend (slope) of time series"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        try:
            slope, _, _, _, _ = stats.linregress(x, values)
            return slope
        except:
            return 0.0
    
    def _calculate_stability(self, values: np.ndarray) -> float:
        """Calculate stability (inverse coefficient of variation)"""
        if len(values) == 0 or np.mean(values) == 0:
            return 0.0
        cv = np.std(values) / np.mean(values)
        return 1.0 / (1.0 + cv)
    
    def _count_peaks(self, values: np.ndarray, threshold: float = 0.1) -> int:
        """Count significant peaks in the signal"""
        if len(values) < 3:
            return 0
        
        peaks = 0
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1] and values[i] > threshold:
                peaks += 1
        return peaks
    
    def _classify_performance(self, video_data: pd.DataFrame) -> str:
        """Classify overall performance based on attention metrics"""
        avg_focus = video_data['focus_level'].mean()
        avg_engagement = video_data['engagement'].mean()
        
        if avg_focus > 0.7 and avg_engagement > 0.7:
            return "High Performance"
        elif avg_focus > 0.5 and avg_engagement > 0.5:
            return "Good Performance"
        elif avg_focus > 0.3 or avg_engagement > 0.3:
            return "Moderate Performance"
        else:
            return "Low Performance"
    
    def _classify_attention_profile(self, video_data: pd.DataFrame) -> str:
        """Classify attention profile based on patterns"""
        focus_std = video_data['focus_level'].std()
        engagement_std = video_data['engagement'].std()
        avg_focus = video_data['focus_level'].mean()
        avg_engagement = video_data['engagement'].mean()
        
        if focus_std < 0.1 and engagement_std < 0.1:
            return "Stable"
        elif focus_std > 0.3 or engagement_std > 0.3:
            return "Variable"
        elif avg_focus > avg_engagement + 0.2:
            return "Focus-Dominant"
        elif avg_engagement > avg_focus + 0.2:
            return "Engagement-Dominant"
        else:
            return "Balanced"
    
    def generate_comprehensive_report(self, output_path: str = "attention_statistics_report.html"):
        """Generate comprehensive HTML report"""
        logger.info("📋 Generating comprehensive attention statistics report...")
        
        # Prepare data for JavaScript
        report_data = {
            'databases': list(self.video_stats.keys()),
            'video_stats': self.video_stats,
            'summary_stats': self._calculate_summary_statistics()
        }
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG Attention Model Statistics Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; }}
        .header h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .controls {{ background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: rgba(255,255,255,0.95); border-radius: 15px; padding: 20px; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .video-table {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; margin-bottom: 25px; }}
        .section-title {{ font-size: 1.5em; font-weight: 600; color: #667eea; margin-bottom: 20px; }}
        .viz-section {{ background: rgba(255,255,255,0.95); border-radius: 20px; padding: 25px; margin-bottom: 25px; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; }}
        .viz-container {{ background: #f8f9fa; border-radius: 10px; padding: 20px; }}
        .performance-tag {{ padding: 5px 10px; border-radius: 15px; color: white; font-size: 0.9em; font-weight: 600; }}
        .high-performance {{ background: #4caf50; }}
        .good-performance {{ background: #8bc34a; }}
        .moderate-performance {{ background: #ff9800; }}
        .low-performance {{ background: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .control-group {{ margin-bottom: 15px; }}
        .control-group select {{ width: 100%; padding: 10px; border: 2px solid #e0e0e0; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-brain"></i> EEG Attention Model Statistics</h1>
            <p>Comprehensive Analysis of Attention Metrics Across Video Sessions</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label><i class="fas fa-database"></i> Database:</label>
                <select id="database-select" onchange="updateDisplay()">
                    {chr(10).join(f'<option value="{db}">{self._format_db_name(db)}</option>' for db in report_data['databases'])}
                </select>
            </div>
        </div>
        
        <div id="summary-stats" class="stats-grid">
            <!-- Dynamic summary stats will be populated here -->
        </div>
        
        <div class="video-table">
            <div class="section-title"><i class="fas fa-table"></i> Video Statistics Table</div>
            <div id="video-stats-table"></div>
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-chart-line"></i> Attention Metrics Visualization</div>
            <div class="viz-grid">
                <div class="viz-container">
                    <div style="font-weight: 600; margin-bottom: 15px; text-align: center;">📊 Focus vs Engagement by Video</div>
                    <div id="focus-engagement-chart"></div>
                </div>
                <div class="viz-container">
                    <div style="font-weight: 600; margin-bottom: 15px; text-align: center;">📈 Performance Distribution</div>
                    <div id="performance-chart"></div>
                </div>
            </div>
        </div>
        
        <div class="viz-section">
            <div class="section-title"><i class="fas fa-wave-square"></i> Brainwave Analysis</div>
            <div class="viz-grid">
                <div class="viz-container">
                    <div style="font-weight: 600; margin-bottom: 15px; text-align: center;">🧠 Brainwave Power by Video</div>
                    <div id="brainwave-chart"></div>
                </div>
                <div class="viz-container">
                    <div style="font-weight: 600; margin-bottom: 15px; text-align: center;">⚡ Attention Stability</div>
                    <div id="stability-chart"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const reportData = {json.dumps(report_data, indent=2, default=str)};
        let currentDb = reportData.databases[0];
        
        document.addEventListener('DOMContentLoaded', function() {{
            updateDisplay();
        }});
        
        function updateDisplay() {{
            currentDb = document.getElementById('database-select').value;
            updateSummaryStats();
            updateVideoTable();
            createFocusEngagementChart();
            createPerformanceChart();
            createBrainwaveChart();
            createStabilityChart();
        }}
        
        function updateSummaryStats() {{
            const stats = reportData.video_stats[currentDb];
            const videos = Object.values(stats);
            
            const avgFocus = videos.reduce((sum, v) => sum + v.focus_mean, 0) / videos.length;
            const avgEngagement = videos.reduce((sum, v) => sum + v.engagement_mean, 0) / videos.length;
            const totalFragments = videos.reduce((sum, v) => sum + v.total_fragments, 0);
            const avgStability = videos.reduce((sum, v) => sum + v.focus_stability, 0) / videos.length;
            
            const summaryHtml = `
                <div class="stat-card">
                    <div class="stat-value">${{(avgFocus * 100).toFixed(1)}}%</div>
                    <div><i class="fas fa-bullseye"></i> Average Focus</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{(avgEngagement * 100).toFixed(1)}}%</div>
                    <div><i class="fas fa-heart"></i> Average Engagement</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{videos.length}}</div>
                    <div><i class="fas fa-video"></i> Videos Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{totalFragments}}</div>
                    <div><i class="fas fa-puzzle-piece"></i> Total Fragments</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{(avgStability * 100).toFixed(1)}}%</div>
                    <div><i class="fas fa-balance-scale"></i> Average Stability</div>
                </div>
            `;
            document.getElementById('summary-stats').innerHTML = summaryHtml;
        }}
        
        function updateVideoTable() {{
            const stats = reportData.video_stats[currentDb];
            
            let tableHtml = `
                <table>
                    <thead>
                        <tr>
                            <th>Video ID</th>
                            <th>Focus Mean</th>
                            <th>Engagement Mean</th>
                            <th>Focus Stability</th>
                            <th>Performance</th>
                            <th>Profile</th>
                            <th>Fragments</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            Object.values(stats).forEach(video => {{
                const performanceClass = video.performance_category.toLowerCase().replace(' ', '-');
                tableHtml += `
                    <tr>
                        <td><strong>${{video.video_short}}</strong></td>
                        <td>${{(video.focus_mean * 100).toFixed(1)}}%</td>
                        <td>${{(video.engagement_mean * 100).toFixed(1)}}%</td>
                        <td>${{(video.focus_stability * 100).toFixed(1)}}%</td>
                        <td><span class="performance-tag ${{performanceClass}}">${{video.performance_category}}</span></td>
                        <td>${{video.attention_profile}}</td>
                        <td>${{video.total_fragments}}</td>
                    </tr>
                `;
            }});
            
            tableHtml += '</tbody></table>';
            document.getElementById('video-stats-table').innerHTML = tableHtml;
        }}
        
        function createFocusEngagementChart() {{
            const stats = reportData.video_stats[currentDb];
            const videos = Object.values(stats);
            
            const trace = {{
                x: videos.map(v => v.focus_mean),
                y: videos.map(v => v.engagement_mean),
                mode: 'markers+text',
                type: 'scatter',
                text: videos.map(v => v.video_short),
                textposition: 'top center',
                marker: {{
                    size: videos.map(v => v.total_fragments / 10 + 8),
                    color: videos.map(v => v.focus_stability),
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: {{title: 'Stability'}}
                }}
            }};
            
            const layout = {{
                xaxis: {{title: 'Focus Level', range: [0, 1]}},
                yaxis: {{title: 'Engagement Level', range: [0, 1]}},
                height: 400
            }};
            
            Plotly.newPlot('focus-engagement-chart', [trace], layout, {{responsive: true}});
        }}
        
        function createPerformanceChart() {{
            const stats = reportData.video_stats[currentDb];
            const videos = Object.values(stats);
            
            const performanceCounts = {{}};
            videos.forEach(v => {{
                performanceCounts[v.performance_category] = (performanceCounts[v.performance_category] || 0) + 1;
            }});
            
            const trace = {{
                labels: Object.keys(performanceCounts),
                values: Object.values(performanceCounts),
                type: 'pie',
                marker: {{
                    colors: ['#4caf50', '#8bc34a', '#ff9800', '#f44336']
                }}
            }};
            
            const layout = {{
                height: 400,
                showlegend: true
            }};
            
            Plotly.newPlot('performance-chart', [trace], layout, {{responsive: true}});
        }}
        
        function createBrainwaveChart() {{
            const stats = reportData.video_stats[currentDb];
            const videos = Object.values(stats);
            
            const traces = [
                {{
                    x: videos.map(v => v.video_short),
                    y: videos.map(v => v.alpha_mean),
                    name: 'Alpha',
                    type: 'bar'
                }},
                {{
                    x: videos.map(v => v.video_short),
                    y: videos.map(v => v.beta_mean),
                    name: 'Beta',
                    type: 'bar'
                }},
                {{
                    x: videos.map(v => v.video_short),
                    y: videos.map(v => v.theta_mean),
                    name: 'Theta',
                    type: 'bar'
                }},
                {{
                    x: videos.map(v => v.video_short),
                    y: videos.map(v => v.gamma_mean),
                    name: 'Gamma',
                    type: 'bar'
                }}
            ];
            
            const layout = {{
                xaxis: {{title: 'Video ID', tickangle: 45}},
                yaxis: {{title: 'Brainwave Power'}},
                height: 400,
                barmode: 'group'
            }};
            
            Plotly.newPlot('brainwave-chart', traces, layout, {{responsive: true}});
        }}
        
        function createStabilityChart() {{
            const stats = reportData.video_stats[currentDb];
            const videos = Object.values(stats);
            
            const trace = {{
                x: videos.map(v => v.video_short),
                y: videos.map(v => v.focus_stability),
                type: 'bar',
                marker: {{color: '#667eea'}},
                name: 'Focus Stability'
            }};
            
            const layout = {{
                xaxis: {{title: 'Video ID', tickangle: 45}},
                yaxis: {{title: 'Stability Score', range: [0, 1]}},
                height: 400
            }};
            
            Plotly.newPlot('stability-chart', [trace], layout, {{responsive: true}});
        }}
    </script>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Comprehensive attention statistics report created: {output_path}")
        return output_path
    
    def _calculate_summary_statistics(self) -> Dict:
        """Calculate summary statistics across all databases"""
        summary = {}
        
        for db_name, video_stats in self.video_stats.items():
            videos = list(video_stats.values())
            
            summary[db_name] = {
                'total_videos': len(videos),
                'avg_focus': np.mean([v['focus_mean'] for v in videos]),
                'avg_engagement': np.mean([v['engagement_mean'] for v in videos]),
                'avg_stability': np.mean([v['focus_stability'] for v in videos]),
                'performance_distribution': self._get_performance_distribution(videos)
            }
        
        return summary
    
    def _get_performance_distribution(self, videos: List[Dict]) -> Dict:
        """Get performance category distribution"""
        distribution = {}
        for video in videos:
            category = video['performance_category']
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _format_db_name(self, db_name: str) -> str:
        """Format database name for display"""
        if '5s' in db_name:
            return "5s Fragments (High Resolution)"
        elif '20s' in db_name:
            return "20s Fragments (Medium Resolution)"
        else:
            return "Full Videos (Global Analysis)"
    
    def export_csv_report(self, output_path: str = "attention_statistics.csv"):
        """Export detailed statistics to CSV"""
        logger.info("📊 Exporting attention statistics to CSV...")
        
        all_stats = []
        for db_name, video_stats in self.video_stats.items():
            for video_id, stats in video_stats.items():
                stats['database'] = db_name
                all_stats.append(stats)
        
        df = pd.DataFrame(all_stats)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ CSV report exported: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Extract attention model statistics")
    parser.add_argument("--db_paths", nargs='+', required=True, help="Database paths")
    parser.add_argument("--output_html", default="attention_statistics_report.html", help="HTML output file")
    parser.add_argument("--output_csv", default="attention_statistics.csv", help="CSV output file")
    
    args = parser.parse_args()
    
    print("🧠 EEG Attention Model Statistics Extractor")
    print("=" * 50)
    
    extractor = AttentionStatsExtractor(args.db_paths)
    
    # Generate reports
    html_path = extractor.generate_comprehensive_report(args.output_html)
    csv_path = extractor.export_csv_report(args.output_csv)
    
    print(f"\n📋 HTML Report: file://{Path(html_path).absolute()}")
    print(f"📊 CSV Export: {Path(csv_path).absolute()}")

if __name__ == "__main__":
    main()
