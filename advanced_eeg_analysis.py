#!/usr/bin/env python3
"""
Advanced EEG Analysis with NeuroLM-L Checkpoint
Performs detailed clustering, classification, and cognitive state analysis
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEEGAnalyzer:
    """Advanced analysis of EEG embeddings with NeuroLM-L"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("./advanced_analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # NeuroLM-L checkpoint
        self.model_name = "Weibang/NeuroLM"
        self.model = None
        self.tokenizer = None
        
    def load_neurolm_checkpoint(self):
        """Load official NeuroLM-L checkpoint from HuggingFace"""
        try:
            logger.info(f"🔧 Loading NeuroLM-L checkpoint: {self.model_name}")
            
            # Try to load the official checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            logger.info("✅ NeuroLM-L checkpoint loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load NeuroLM-L checkpoint: {e}")
            logger.info("💡 Note: You may need authentication for the checkpoint")
            return False
    
    def analyze_dataset_structure(self):
        """Analyze the structure of your EEG dataset"""
        logger.info("📊 Analyzing dataset structure")
        
        csv_files = list(self.data_dir.glob("lsl_stream_*.csv"))
        
        dataset_info = []
        for csv_file in csv_files:
            try:
                # Read header info
                with open(csv_file, 'r') as f:
                    header_lines = []
                    for line in f:
                        if line.startswith('#'):
                            header_lines.append(line.strip())
                        else:
                            break
                
                # Read data
                df = pd.read_csv(csv_file, comment='#')
                
                # Extract info
                info = {
                    'session_id': csv_file.stem,
                    'file_size_mb': csv_file.stat().st_size / (1024 * 1024),
                    'n_samples': len(df),
                    'n_channels': len([col for col in df.columns if col.startswith('channel_')]),
                    'duration_seconds': df['unix_timestamp'].iloc[-1] - df['unix_timestamp'].iloc[0],
                    'sampling_rate_est': len(df) / (df['unix_timestamp'].iloc[-1] - df['unix_timestamp'].iloc[0]),
                    'header_info': header_lines
                }
                
                dataset_info.append(info)
                
            except Exception as e:
                logger.warning(f"⚠️ Error analyzing {csv_file}: {e}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(dataset_info)
        summary_df.to_csv(self.results_dir / "dataset_summary.csv", index=False)
        
        # Print summary
        print("\n📊 DATASET ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"📁 Total sessions: {len(dataset_info)}")
        print(f"⏱️ Total duration: {summary_df['duration_seconds'].sum()/3600:.1f} hours")
        print(f"📊 Average samples per session: {summary_df['n_samples'].mean():.0f}")
        print(f"🎛️ Channels per session: {summary_df['n_channels'].iloc[0]}")
        print(f"📈 Estimated sampling rate: {summary_df['sampling_rate_est'].mean():.1f} Hz")
        print(f"💾 Total data size: {summary_df['file_size_mb'].sum():.1f} MB")
        
        return summary_df
    
    def extract_cognitive_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract traditional cognitive features from EEG for comparison
        
        Args:
            eeg_data: EEG data [n_samples, n_channels]
            
        Returns:
            features: Dictionary of cognitive features
        """
        features = {}
        
        # Basic statistical features
        features['mean_power'] = np.mean(eeg_data)
        features['std_power'] = np.std(eeg_data)
        features['max_power'] = np.max(eeg_data)
        features['min_power'] = np.min(eeg_data)
        
        # Channel-wise features
        for i in range(eeg_data.shape[1]):
            channel_data = eeg_data[:, i]
            features[f'ch{i+1}_mean'] = np.mean(channel_data)
            features[f'ch{i+1}_std'] = np.std(channel_data)
            features[f'ch{i+1}_power'] = np.sum(channel_data ** 2)
        
        # Cross-channel correlations
        corr_matrix = np.corrcoef(eeg_data.T)
        features['avg_correlation'] = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        features['max_correlation'] = np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        
        return features
    
    def compare_clustering_methods(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compare different clustering methods"""
        logger.info("🎯 Comparing clustering methods")
        
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        results = {}
        
        # K-Means with different k values
        k_range = range(2, min(8, len(embeddings)))
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings_scaled)
            score = silhouette_score(embeddings_scaled, labels)
            silhouette_scores.append(score)
        
        results['kmeans_scores'] = dict(zip(k_range, silhouette_scores))
        results['optimal_k'] = k_range[np.argmax(silhouette_scores)]
        
        # DBSCAN with different parameters
        eps_values = [0.3, 0.5, 0.7, 1.0]
        dbscan_results = []
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            labels = dbscan.fit_predict(embeddings_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            dbscan_results.append({
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'labels': labels
            })
        
        results['dbscan_results'] = dbscan_results
        
        return results
    
    def create_comprehensive_report(self, embeddings: np.ndarray, 
                                  cluster_results: Dict[str, Any],
                                  session_metadata: List[Dict]) -> str:
        """Create a comprehensive analysis report"""
        
        report = []
        report.append("# 🧠 NeuroLM EEG Video Classification Report")
        report.append("=" * 60)
        report.append("")
        
        # Dataset overview
        report.append("## 📊 Dataset Overview")
        report.append(f"- **Total sessions**: {len(session_metadata)}")
        report.append(f"- **Embedding dimension**: {embeddings.shape[1]}")
        report.append(f"- **Total duration**: {sum(meta['duration_minutes'] for meta in session_metadata):.1f} minutes")
        report.append("")
        
        # Clustering results
        optimal_k = cluster_results.get('optimal_k', 3)
        optimal_score = cluster_results['kmeans_scores'][optimal_k]
        
        report.append("## 🎯 Clustering Results")
        report.append(f"- **Optimal K**: {optimal_k}")
        report.append(f"- **Silhouette Score**: {optimal_score:.3f}")
        report.append("")
        
        # Session analysis by cluster
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(StandardScaler().fit_transform(embeddings))
        
        report.append("## 📋 Sessions by Cluster")
        for cluster_id in range(optimal_k):
            mask = labels == cluster_id
            cluster_sessions = [session_metadata[i]['session_id'] for i in np.where(mask)[0]]
            avg_duration = np.mean([session_metadata[i]['duration_minutes'] for i in np.where(mask)[0]])
            
            report.append(f"### Cluster {cluster_id}")
            report.append(f"- **Sessions**: {', '.join(cluster_sessions)}")
            report.append(f"- **Average duration**: {avg_duration:.1f} minutes")
            report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open(self.results_dir / "analysis_report.md", 'w') as f:
            f.write(report_text)
        
        return report_text

def run_analysis():
    """Run the complete EEG analysis pipeline"""
    analyzer = AdvancedEEGAnalyzer("/Users/e.baena/Desktop/data")
    
    # Analyze dataset structure first
    print("🔍 Analyzing dataset structure...")
    dataset_summary = analyzer.analyze_dataset_structure()
    
    print("\n📋 Dataset loaded successfully!")
    print("🚀 Ready to run NeuroLM classification")
    print("\nNext steps:")
    print("1. Run: python eeg_video_classification_system.py")
    print("2. Check results in: ./eeg_classification_results/")
    print("3. Open interactive plot: eeg_classification_analysis.html")

if __name__ == "__main__":
    run_analysis()
