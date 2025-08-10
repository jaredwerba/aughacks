#!/usr/bin/env python3
"""
EEG Video Classification & Clustering System using NeuroLM-L
Analyzes video/EEG dataset using official NeuroLM HuggingFace checkpoint
"""

import asyncio
import csv
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from transformers import AutoTokenizer, AutoModel
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our NeuroLM components
from neurolm_tokenizer import NeuroLMTokenizer, NeuroTokenizerConfig
from neurolm_attention_model import NeuroLMAttentionModel, AttentionModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroLMEEGClassifier:
    """EEG Video Classification System using NeuroLM-L HuggingFace Checkpoint"""
    
    def __init__(self, data_dir: str, checkpoint_name: str = "Weibang/NeuroLM"):
        self.data_dir = Path(data_dir)
        self.checkpoint_name = checkpoint_name
        self.output_dir = Path("./eeg_classification_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Dataset information
        self.sessions = []
        self.embeddings = []
        self.session_metadata = []
        
        # NeuroLM components
        self.tokenizer = None
        self.model = None
        self.hf_model = None
        
        # Results
        self.cluster_results = {}
        
    async def initialize_neurolm_checkpoint(self) -> bool:
        """Initialize NeuroLM-L checkpoint from HuggingFace"""
        try:
            logger.info(f"🔧 Loading NeuroLM checkpoint: {self.checkpoint_name}")
            
            # Try HuggingFace checkpoint first
            try:
                self.hf_model = AutoModel.from_pretrained(
                    self.checkpoint_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                logger.info("✅ HuggingFace NeuroLM-L checkpoint loaded successfully")
                return True
            except Exception as hf_error:
                logger.warning(f"⚠️ HuggingFace checkpoint failed: {hf_error}")
                
                # Fallback to local implementation
                tokenizer_config = NeuroTokenizerConfig(
                    sampling_rate=250, window_size=200, n_channels=6,
                    n_embed=8192, embed_dim=128
                )
                
                attention_config = AttentionModelConfig(
                    vocab_size=8192, n_layer=12, n_head=12, n_embd=768
                )
                
                self.tokenizer = NeuroLMTokenizer(tokenizer_config)
                self.model = NeuroLMAttentionModel(attention_config)
                logger.info("✅ Local NeuroLM components initialized")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize NeuroLM: {e}")
            return False
    
    def load_eeg_sessions(self) -> List[Dict]:
        """Load all EEG sessions from CSV files"""
        logger.info(f"📂 Loading EEG sessions from {self.data_dir}")
        
        csv_files = list(self.data_dir.glob("lsl_stream_*.csv"))
        logger.info(f"📁 Found {len(csv_files)} EEG session files")
        
        sessions = []
        for csv_file in sorted(csv_files):
            try:
                # Read CSV with proper handling
                df = pd.read_csv(csv_file, comment='#')
                
                # Extract EEG data (skip timestamp columns)
                eeg_columns = [col for col in df.columns if col.startswith('channel_')]
                eeg_data = df[eeg_columns].values
                timestamps = df['unix_timestamp'].values
                
                session_info = {
                    'session_id': csv_file.stem,
                    'duration_minutes': (timestamps[-1] - timestamps[0]) / 60,
                    'n_samples': len(eeg_data),
                    'n_channels': len(eeg_columns),
                    'eeg_data': eeg_data,
                    'timestamps': timestamps,
                    'file_size_mb': csv_file.stat().st_size / (1024 * 1024)
                }
                
                sessions.append(session_info)
                logger.info(f"✅ {session_info['session_id']}: "
                          f"{session_info['duration_minutes']:.1f}min, "
                          f"{session_info['file_size_mb']:.1f}MB")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {csv_file}: {e}")
                continue
        
        self.sessions = sessions
        return sessions
    
    async def extract_embeddings_per_session(self, session: Dict) -> np.ndarray:
        """Extract NeuroLM embeddings for a single EEG session"""
        logger.info(f"🧠 Extracting embeddings: {session['session_id']}")
        
        eeg_data = session['eeg_data']  # [n_samples, n_channels]
        n_samples, n_channels = eeg_data.shape
        
        # Windowing configuration
        window_size = 500  # 2 seconds at 250Hz
        overlap = 250      # 50% overlap
        step_size = window_size - overlap
        
        # Extract windows
        windows = []
        for start_idx in range(0, n_samples - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = eeg_data[start_idx:end_idx].T  # [n_channels, window_size]
            windows.append(window)
        
        if not windows:
            return np.array([])
        
        # Process through NeuroLM
        session_embeddings = []
        channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8']
        
        for window in windows:
            try:
                window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                
                if self.hf_model is not None:
                    # Use HuggingFace checkpoint
                    with torch.no_grad():
                        outputs = self.hf_model(window_tensor)
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                else:
                    # Use local NeuroLM
                    with torch.no_grad():
                        tokenizer_output = self.tokenizer.forward(window_tensor.squeeze(0), channel_names)
                        tokens = tokenizer_output.get('tokens', tokenizer_output.get('quantized'))
                        predictions = self.model.forward(tokens)
                        
                        # Create embedding from predictions
                        if isinstance(predictions, dict):
                            attention_logits = predictions.get('attention_logits', torch.zeros(1, 3))
                            engagement_logits = predictions.get('engagement_logits', torch.zeros(1, 3))
                            embedding = torch.cat([attention_logits.flatten(), engagement_logits.flatten()]).cpu().numpy()
                        else:
                            embedding = predictions.flatten().cpu().numpy()
                
                session_embeddings.append(embedding)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process window: {e}")
                continue
        
        if session_embeddings:
            # Average embeddings for session-level representation
            session_embedding = np.mean(session_embeddings, axis=0)
            logger.info(f"✅ Session embedding shape: {session_embedding.shape}")
            return session_embedding
        else:
            return np.array([])
    
    async def process_all_sessions(self) -> Dict[str, np.ndarray]:
        """Process all sessions and extract embeddings"""
        logger.info("🚀 Starting embedding extraction for all sessions")
        
        session_embeddings = {}
        
        for session in self.sessions:
            embedding = await self.extract_embeddings_per_session(session)
            
            if embedding.size > 0:
                session_embeddings[session['session_id']] = embedding
                self.session_metadata.append({
                    'session_id': session['session_id'],
                    'duration_minutes': session['duration_minutes'],
                    'n_samples': session['n_samples'],
                    'file_size_mb': session['file_size_mb']
                })
        
        logger.info(f"✅ Extracted embeddings for {len(session_embeddings)} sessions")
        self.embeddings = session_embeddings
        return session_embeddings
    
    def perform_clustering(self, embeddings_matrix: np.ndarray) -> Dict[str, Any]:
        """Perform clustering analysis on embeddings"""
        logger.info("🎯 Performing clustering analysis")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)
        
        results = {}
        
        # K-Means clustering with different k values
        logger.info("🔵 Running K-Means clustering")
        kmeans_results = {}
        silhouette_scores = {}
        
        for k in range(2, min(8, len(embeddings_matrix))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_scaled)
            
            silhouette_avg = silhouette_score(embeddings_scaled, labels)
            calinski_score = calinski_harabasz_score(embeddings_scaled, labels)
            
            kmeans_results[k] = {
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'silhouette_score': silhouette_avg,
                'calinski_score': calinski_score
            }
            silhouette_scores[k] = silhouette_avg
            
            logger.info(f"  K={k}: Silhouette={silhouette_avg:.3f}")
        
        # Find optimal K
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        results['kmeans'] = kmeans_results
        results['optimal_k'] = optimal_k
        
        # DBSCAN clustering
        logger.info("🟡 Running DBSCAN clustering")
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(embeddings_scaled)
        
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        results['dbscan'] = {
            'labels': dbscan_labels,
            'n_clusters': n_clusters_dbscan,
            'n_noise': n_noise
        }
        
        logger.info(f"  DBSCAN: {n_clusters_dbscan} clusters, {n_noise} noise points")
        
        self.cluster_results = results
        return results
    
    def create_visualizations(self, embeddings_matrix: np.ndarray) -> None:
        """Create comprehensive visualizations"""
        logger.info("📊 Creating visualizations")
        
        # Dimensionality reduction
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(embeddings_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_matrix)-1))
        tsne_result = tsne.fit_transform(embeddings_scaled)
        
        # Get cluster labels
        optimal_k = self.cluster_results.get('optimal_k', 3)
        labels = self.cluster_results['kmeans'][optimal_k]['labels']
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['PCA Clustering', 't-SNE Clustering', 
                          'Session Metadata', 'Cluster Quality'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Session IDs for labeling
        session_ids = [meta['session_id'] for meta in self.session_metadata]
        colors = px.colors.qualitative.Set3
        
        # PCA plot
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=pca_result[mask, 0], y=pca_result[mask, 1],
                    mode='markers+text',
                    marker=dict(color=colors[cluster_id % len(colors)], size=10),
                    text=[session_ids[j] for j in np.where(mask)[0]],
                    name=f'Cluster {cluster_id}',
                    showlegend=True
                ), row=1, col=1
            )
        
        # t-SNE plot
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=tsne_result[mask, 0], y=tsne_result[mask, 1],
                    mode='markers+text',
                    marker=dict(color=colors[cluster_id % len(colors)], size=10),
                    text=[session_ids[j] for j in np.where(mask)[0]],
                    name=f'Cluster {cluster_id}',
                    showlegend=False
                ), row=1, col=2
            )
        
        # Session metadata
        durations = [meta['duration_minutes'] for meta in self.session_metadata]
        file_sizes = [meta['file_size_mb'] for meta in self.session_metadata]
        
        fig.add_trace(
            go.Scatter(
                x=durations, y=file_sizes,
                mode='markers+text',
                marker=dict(color=[colors[label % len(colors)] for label in labels], size=12),
                text=session_ids,
                name='Sessions',
                showlegend=False
            ), row=2, col=1
        )
        
        # Cluster quality metrics
        k_values = list(self.cluster_results['kmeans'].keys())
        silhouette_values = [self.cluster_results['kmeans'][k]['silhouette_score'] for k in k_values]
        
        fig.add_trace(
            go.Bar(
                x=[f'K={k}' for k in k_values],
                y=silhouette_values,
                name='Silhouette Score',
                showlegend=False,
                marker_color='lightblue'
            ), row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="🧠 NeuroLM EEG Video Classification Analysis",
            height=800,
            template="plotly_dark"
        )
        
        # Save interactive plot
        output_file = self.output_dir / "eeg_classification_analysis.html"
        fig.write_html(str(output_file))
        logger.info(f"💾 Visualization saved: {output_file}")
    
    def save_results(self, embeddings_matrix: np.ndarray) -> None:
        """Save all results to files"""
        logger.info("💾 Saving results")
        
        # Save embeddings
        embeddings_df = pd.DataFrame(embeddings_matrix)
        embeddings_df.index = [meta['session_id'] for meta in self.session_metadata]
        embeddings_df.to_csv(self.output_dir / "session_embeddings.csv")
        
        # Save metadata with clusters
        metadata_df = pd.DataFrame(self.session_metadata)
        optimal_k = self.cluster_results.get('optimal_k', 3)
        metadata_df['cluster_label'] = self.cluster_results['kmeans'][optimal_k]['labels']
        metadata_df.to_csv(self.output_dir / "session_metadata_with_clusters.csv", index=False)
        
        # Save cluster results
        with open(self.output_dir / "cluster_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in self.cluster_results.items():
                if key == 'kmeans':
                    serializable_results[key] = {}
                    for k, k_results in value.items():
                        serializable_results[key][k] = {
                            'labels': k_results['labels'].tolist(),
                            'silhouette_score': float(k_results['silhouette_score']),
                            'calinski_score': float(k_results['calinski_score'])
                        }
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📁 Results saved to {self.output_dir}")

async def main():
    """Main function to run EEG classification analysis"""
    print("🧠 NeuroLM EEG Video Classification & Clustering System")
    print("=" * 60)
    
    # Initialize classifier
    data_dir = "/Users/e.baena/Desktop/data"
    classifier = NeuroLMEEGClassifier(data_dir)
    
    # Step 1: Initialize NeuroLM
    logger.info("🔧 Step 1: Initializing NeuroLM checkpoint")
    success = await classifier.initialize_neurolm_checkpoint()
    if not success:
        logger.error("❌ Failed to initialize NeuroLM")
        return
    
    # Step 2: Load EEG sessions
    logger.info("📂 Step 2: Loading EEG sessions")
    sessions = classifier.load_eeg_sessions()
    if not sessions:
        logger.error("❌ No EEG sessions found")
        return
    
    # Step 3: Extract embeddings
    logger.info("🧠 Step 3: Extracting NeuroLM embeddings")
    embeddings_dict = await classifier.process_all_sessions()
    
    if not embeddings_dict:
        logger.error("❌ No embeddings extracted")
        return
    
    # Convert to matrix
    embeddings_matrix = np.array(list(embeddings_dict.values()))
    logger.info(f"📊 Embeddings matrix shape: {embeddings_matrix.shape}")
    
    # Step 4: Clustering analysis
    logger.info("🎯 Step 4: Performing clustering analysis")
    cluster_results = classifier.perform_clustering(embeddings_matrix)
    
    # Step 5: Create visualizations
    logger.info("📊 Step 5: Creating visualizations")
    classifier.create_visualizations(embeddings_matrix)
    
    # Step 6: Save results
    logger.info("💾 Step 6: Saving results")
    classifier.save_results(embeddings_matrix)
    
    # Summary
    optimal_k = cluster_results.get('optimal_k', 3)
    optimal_silhouette = cluster_results['kmeans'][optimal_k]['silhouette_score']
    
    print("\n🎯 CLASSIFICATION SUMMARY")
    print("=" * 40)
    print(f"📁 Sessions analyzed: {len(embeddings_dict)}")
    print(f"🧠 Embedding dimension: {embeddings_matrix.shape[1]}")
    print(f"🎯 Optimal clusters (K-means): {optimal_k}")
    print(f"📊 Silhouette score: {optimal_silhouette:.3f}")
    print(f"💾 Results saved to: {classifier.output_dir}")
    print(f"📈 Interactive plot: {classifier.output_dir}/eeg_classification_analysis.html")

if __name__ == "__main__":
    asyncio.run(main())
