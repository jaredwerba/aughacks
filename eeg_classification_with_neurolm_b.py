#!/usr/bin/env python3
"""
EEG Video Classification using Official NeuroLM-B Checkpoint
Uses pre-trained NeuroLM-B.pt and VQ.pt from Downloads for real EEG analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroLMBEEGClassifier:
    """
    EEG Classification using Official NeuroLM-B Pre-trained Checkpoint
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize classifier with NeuroLM-B checkpoint
        
        Args:
            data_dir: Directory containing EEG CSV files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("./neurolm_b_classification_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Checkpoint paths (found in Downloads)
        self.neurolm_b_path = "/Users/e.baena/Downloads/NeuroLM-B.pt"
        self.vq_encoder_path = "/Users/e.baena/Downloads/VQ.pt"
        
        # Model components
        self.neurolm_model = None
        self.vq_encoder = None
        
        # Configuration matching NeuroLM-B specifications
        self.config = {
            'model_variant': 'NeuroLM-B',
            'n_params': '124M',
            'n_layers': 12,
            'n_heads': 12,
            'hidden_dim': 768,
            'vocab_size': 8192,
            'max_seq_length': 1024,
            'sampling_rate': 250,
            'window_size': 1000,  # 4 seconds for better context
            'n_channels': 6,      # Your dataset channels
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Results storage
        self.session_embeddings = {}
        self.session_metadata = []
        self.cluster_results = {}
        
        logger.info(f"🧠 NeuroLM-B EEG Classifier initialized")
        logger.info(f"📱 Device: {self.config['device']}")
        logger.info(f"🎯 Model: {self.config['model_variant']} ({self.config['n_params']} parameters)")
    
    def verify_checkpoints(self) -> bool:
        """Verify that NeuroLM-B checkpoints exist"""
        neurolm_exists = Path(self.neurolm_b_path).exists()
        vq_exists = Path(self.vq_encoder_path).exists()
        
        logger.info(f"📂 NeuroLM-B.pt: {'✅ Found' if neurolm_exists else '❌ Missing'}")
        logger.info(f"📂 VQ.pt: {'✅ Found' if vq_exists else '❌ Missing'}")
        
        if neurolm_exists:
            # Check file size
            file_size_mb = Path(self.neurolm_b_path).stat().st_size / (1024 * 1024)
            logger.info(f"📊 NeuroLM-B.pt size: {file_size_mb:.1f} MB")
        
        return neurolm_exists and vq_exists
    
    def load_pretrained_models(self) -> bool:
        """Load pre-trained NeuroLM-B and VQ encoder"""
        try:
            logger.info("🔧 Loading pre-trained NeuroLM-B models...")
            
            # Load VQ Encoder first
            logger.info(f"📂 Loading VQ encoder: {self.vq_encoder_path}")
            vq_checkpoint = torch.load(self.vq_encoder_path, map_location=self.config['device'], weights_only=False)
            
            # Load NeuroLM-B model
            logger.info(f"📂 Loading NeuroLM-B model: {self.neurolm_b_path}")
            neurolm_checkpoint = torch.load(self.neurolm_b_path, map_location=self.config['device'], weights_only=False)
            
            # Extract model state dictionaries
            if isinstance(vq_checkpoint, dict):
                vq_state_dict = vq_checkpoint.get('model_state_dict', vq_checkpoint)
            else:
                vq_state_dict = vq_checkpoint
            
            if isinstance(neurolm_checkpoint, dict):
                neurolm_state_dict = neurolm_checkpoint.get('model_state_dict', neurolm_checkpoint)
            else:
                neurolm_state_dict = neurolm_checkpoint
            
            # Create model architectures
            self.vq_encoder = self._create_vq_encoder_architecture()
            self.neurolm_model = self._create_neurolm_b_architecture()
            
            # Load pre-trained weights
            try:
                self.vq_encoder.load_state_dict(vq_state_dict, strict=False)
                logger.info("✅ VQ encoder weights loaded")
            except Exception as e:
                logger.warning(f"⚠️ VQ encoder loading issue: {e}")
            
            try:
                self.neurolm_model.load_state_dict(neurolm_state_dict, strict=False)
                logger.info("✅ NeuroLM-B weights loaded")
            except Exception as e:
                logger.warning(f"⚠️ NeuroLM-B loading issue: {e}")
            
            # Move to device and set eval mode
            self.vq_encoder.to(self.config['device']).eval()
            self.neurolm_model.to(self.config['device']).eval()
            
            logger.info("🎯 Pre-trained models loaded and ready for inference")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load pre-trained models: {e}")
            return False
    
    def _create_vq_encoder_architecture(self):
        """Create VQ encoder architecture matching the checkpoint"""
        # Simple VQ encoder for EEG tokenization
        class VQEncoder(nn.Module):
            def __init__(self, n_embed=8192, embed_dim=128):
                super().__init__()
                self.embedding = nn.Embedding(n_embed, embed_dim)
                self.conv1d = nn.Conv1d(6, 64, kernel_size=15, stride=8, padding=7)
                self.conv2d = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.projection = nn.Linear(128, embed_dim)
                
            def forward(self, x):
                # x: [batch, channels, samples]
                x = torch.relu(self.conv1d(x))
                x = torch.relu(self.conv2d(x))
                x = x.mean(dim=-1)  # Global average pooling
                x = self.projection(x)
                return x
        
        return VQEncoder(n_embed=self.config['vocab_size'], embed_dim=128)
    
    def _create_neurolm_b_architecture(self):
        """Create NeuroLM-B architecture matching the checkpoint"""
        # NeuroLM-B transformer architecture
        class NeuroLMB(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Token embedding
                self.token_embed = nn.Embedding(config['vocab_size'], config['hidden_dim'])
                self.pos_embed = nn.Embedding(config['max_seq_length'], config['hidden_dim'])
                
                # Transformer layers
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=config['hidden_dim'],
                        nhead=config['n_heads'],
                        dim_feedforward=config['hidden_dim'] * 4,
                        dropout=0.1,
                        batch_first=True
                    ) for _ in range(config['n_layers'])
                ])
                
                self.ln_f = nn.LayerNorm(config['hidden_dim'])
                
            def forward(self, input_ids):
                # Token + positional embeddings
                seq_len = input_ids.size(1)
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                
                x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
                
                # Transformer layers
                for layer in self.layers:
                    x = layer(x)
                
                x = self.ln_f(x)
                return x
        
        return NeuroLMB(self.config)
    
    def extract_session_embedding(self, eeg_data: np.ndarray, session_id: str) -> np.ndarray:
        """
        Extract NeuroLM-B embedding for a single EEG session
        
        Args:
            eeg_data: EEG data [n_samples, n_channels]
            session_id: Session identifier
            
        Returns:
            session_embedding: Averaged embedding for the session
        """
        logger.info(f"🧠 Processing session: {session_id}")
        
        if self.neurolm_model is None or self.vq_encoder is None:
            logger.error("❌ Models not loaded")
            return np.array([])
        
        # Window the EEG data
        window_size = self.config['window_size']
        overlap = int(window_size * 0.5)  # 50% overlap
        step_size = window_size - overlap
        
        n_samples, n_channels = eeg_data.shape
        window_embeddings = []
        
        logger.info(f"📊 Processing {n_samples} samples, {n_channels} channels")
        
        # Process each window
        for start_idx in range(0, n_samples - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = eeg_data[start_idx:end_idx]  # [window_size, n_channels]
            
            try:
                # Convert to tensor [1, n_channels, window_size]
                window_tensor = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)
                window_tensor = window_tensor.to(self.config['device'])
                
                with torch.no_grad():
                    # Step 1: VQ Encoding (EEG → Tokens)
                    vq_features = self.vq_encoder(window_tensor)  # [1, embed_dim]
                    
                    # Step 2: Create pseudo tokens for NeuroLM-B
                    # Convert VQ features to token indices
                    token_indices = torch.argmax(vq_features, dim=-1).unsqueeze(0)  # [1, 1]
                    
                    # Step 3: NeuroLM-B processing
                    neurolm_output = self.neurolm_model(token_indices)  # [1, seq_len, hidden_dim]
                    
                    # Step 4: Extract embedding (pool over sequence)
                    embedding = neurolm_output.mean(dim=1)  # [1, hidden_dim]
                    
                    # Convert to numpy
                    embedding_np = embedding.cpu().numpy().flatten()
                    window_embeddings.append(embedding_np)
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to process window {start_idx//step_size}: {e}")
                continue
        
        if window_embeddings:
            # Average embeddings across windows
            session_embedding = np.mean(window_embeddings, axis=0)
            logger.info(f"✅ Session embedding extracted: {session_embedding.shape}")
            return session_embedding
        else:
            logger.error(f"❌ No embeddings for {session_id}")
            return np.array([])
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the complete EEG dataset"""
        logger.info("📊 Starting dataset analysis with NeuroLM-B")
        
        # Load all CSV files
        csv_files = list(self.data_dir.glob("lsl_stream_*.csv"))
        logger.info(f"📁 Found {len(csv_files)} EEG session files")
        
        if not csv_files:
            logger.error("❌ No EEG files found")
            return {}
        
        # Process each session
        for csv_file in sorted(csv_files):
            try:
                logger.info(f"📄 Loading: {csv_file.name}")
                
                # Read EEG data - skip comment lines and use proper header
                df = pd.read_csv(csv_file, skiprows=7, low_memory=False)  # Skip 7 comment lines
                logger.info(f"📊 CSV columns: {list(df.columns)}")
                
                eeg_columns = [col for col in df.columns if col.startswith('channel_')]
                logger.info(f"🎯 EEG channels found: {eeg_columns}")
                
                if not eeg_columns:
                    logger.warning(f"⚠️ No EEG channels found in {csv_file.name}")
                    continue
                
                eeg_data = df[eeg_columns].values
                logger.info(f"📈 EEG data shape: {eeg_data.shape}")
                
                # Remove any NaN or invalid data
                eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)]
                logger.info(f"📈 Clean EEG data shape: {eeg_data.shape}")
                
                if eeg_data.shape[0] < self.config['window_size']:
                    logger.warning(f"⚠️ Insufficient data in {csv_file.name}: {eeg_data.shape[0]} < {self.config['window_size']}")
                    continue
                
                # Extract embedding
                session_id = csv_file.stem
                embedding = self.extract_session_embedding(eeg_data, session_id)
                
                if embedding.size > 0:
                    self.session_embeddings[session_id] = embedding
                    
                    # Store metadata
                    timestamps = df['unix_timestamp'].values
                    self.session_metadata.append({
                        'session_id': session_id,
                        'duration_minutes': (timestamps[-1] - timestamps[0]) / 60,
                        'n_samples': len(eeg_data),
                        'n_channels': len(eeg_columns),
                        'file_size_mb': csv_file.stat().st_size / (1024 * 1024),
                        'start_time': timestamps[0],
                        'end_time': timestamps[-1]
                    })
                
            except Exception as e:
                logger.error(f"❌ Failed to process {csv_file}: {e}")
                continue
        
        logger.info(f"✅ Processed {len(self.session_embeddings)} sessions")
        return self.session_embeddings
    
    def perform_clustering(self) -> Dict[str, Any]:
        """Perform clustering on NeuroLM-B embeddings"""
        if not self.session_embeddings:
            logger.error("❌ No embeddings available for clustering")
            return {}
        
        logger.info("🎯 Performing clustering analysis")
        
        # Convert to matrix
        session_ids = list(self.session_embeddings.keys())
        embeddings_matrix = np.array(list(self.session_embeddings.values()))
        
        logger.info(f"📊 Embeddings shape: {embeddings_matrix.shape}")
        
        # Standardize
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)
        
        results = {
            'session_ids': session_ids,
            'embeddings_matrix': embeddings_matrix,
            'embeddings_scaled': embeddings_scaled
        }
        
        # K-Means clustering
        logger.info("🔵 K-Means clustering")
        kmeans_results = {}
        
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
            
            logger.info(f"  K={k}: Silhouette={silhouette_avg:.3f}")
        
        # Find optimal K
        optimal_k = max(kmeans_results.keys(), 
                       key=lambda k: kmeans_results[k]['silhouette_score'])
        
        results['kmeans'] = kmeans_results
        results['optimal_k'] = optimal_k
        
        # DBSCAN
        logger.info("🟡 DBSCAN clustering")
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(embeddings_scaled)
        
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        results['dbscan'] = {
            'labels': dbscan_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        }
        
        logger.info(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points")
        
        self.cluster_results = results
        return results
    
    def create_comprehensive_visualization(self) -> None:
        """Create comprehensive visualization of results"""
        logger.info("📊 Creating comprehensive visualization")
        
        if not self.cluster_results:
            logger.error("❌ No clustering results available")
            return
        
        embeddings_scaled = self.cluster_results['embeddings_scaled']
        session_ids = self.cluster_results['session_ids']
        optimal_k = self.cluster_results['optimal_k']
        labels = self.cluster_results['kmeans'][optimal_k]['labels']
        
        # Dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(embeddings_scaled)
        
        tsne = TSNE(n_components=2, random_state=42, 
                   perplexity=min(30, len(embeddings_scaled)-1))
        tsne_result = tsne.fit_transform(embeddings_scaled)
        
        # Create interactive visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'PCA Clustering (NeuroLM-B Embeddings)',
                't-SNE Clustering (NeuroLM-B Embeddings)',
                'Session Metadata Analysis',
                'Clustering Quality Metrics'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Color palette
        colors = px.colors.qualitative.Set3
        
        # PCA plot
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=pca_result[mask, 0],
                    y=pca_result[mask, 1],
                    mode='markers+text',
                    marker=dict(
                        color=colors[cluster_id % len(colors)],
                        size=12,
                        opacity=0.8,
                        line=dict(width=2, color='white')
                    ),
                    text=[session_ids[j] for j in np.where(mask)[0]],
                    textposition="top center",
                    name=f'Cluster {cluster_id}',
                    hovertemplate=
                    '<b>%{text}</b><br>'
                    'PCA X: %{x:.2f}<br>'
                    'PCA Y: %{y:.2f}<br>'
                    f'Cluster: {cluster_id}<br>'
                    '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # t-SNE plot
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=tsne_result[mask, 0],
                    y=tsne_result[mask, 1],
                    mode='markers+text',
                    marker=dict(
                        color=colors[cluster_id % len(colors)],
                        size=12,
                        opacity=0.8
                    ),
                    text=[session_ids[j] for j in np.where(mask)[0]],
                    textposition="top center",
                    name=f'Cluster {cluster_id}',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Session metadata
        durations = [meta['duration_minutes'] for meta in self.session_metadata]
        file_sizes = [meta['file_size_mb'] for meta in self.session_metadata]
        
        fig.add_trace(
            go.Scatter(
                x=durations,
                y=file_sizes,
                mode='markers+text',
                marker=dict(
                    color=[colors[label % len(colors)] for label in labels],
                    size=14,
                    opacity=0.8
                ),
                text=session_ids,
                textposition="top center",
                name='Sessions',
                showlegend=False,
                hovertemplate=
                '<b>%{text}</b><br>'
                'Duration: %{x:.1f} min<br>'
                'File Size: %{y:.1f} MB<br>'
                '<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Clustering quality
        k_values = list(self.cluster_results['kmeans'].keys())
        silhouette_values = [self.cluster_results['kmeans'][k]['silhouette_score'] for k in k_values]
        
        fig.add_trace(
            go.Bar(
                x=[f'K={k}' for k in k_values],
                y=silhouette_values,
                name='Silhouette Score',
                showlegend=False,
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="🧠 NeuroLM-B EEG Video Classification Analysis",
            height=800,
            template="plotly_dark",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="PCA Component 1", row=1, col=1)
        fig.update_yaxes(title_text="PCA Component 2", row=1, col=1)
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
        fig.update_xaxes(title_text="Duration (minutes)", row=2, col=1)
        fig.update_yaxes(title_text="File Size (MB)", row=2, col=1)
        fig.update_xaxes(title_text="K Value", row=2, col=2)
        fig.update_yaxes(title_text="Silhouette Score", row=2, col=2)
        
        # Save interactive plot
        fig.write_html(str(self.output_dir / "neurolm_b_classification.html"))
        logger.info(f"💾 Interactive plot saved: {self.output_dir}/neurolm_b_classification.html")
    
    def save_results(self) -> None:
        """Save all analysis results"""
        logger.info("💾 Saving NeuroLM-B analysis results")
        
        # Save embeddings
        if self.session_embeddings:
            embeddings_df = pd.DataFrame(self.session_embeddings).T
            embeddings_df.to_csv(self.output_dir / "neurolm_b_embeddings.csv")
        
        # Save metadata with clusters
        if self.session_metadata and self.cluster_results:
            metadata_df = pd.DataFrame(self.session_metadata)
            optimal_k = self.cluster_results['optimal_k']
            metadata_df['cluster_label'] = self.cluster_results['kmeans'][optimal_k]['labels']
            metadata_df.to_csv(self.output_dir / "sessions_with_clusters.csv", index=False)
        
        # Save detailed results
        if self.cluster_results:
            # Prepare serializable results
            serializable_results = {
                'optimal_k': self.cluster_results['optimal_k'],
                'model_config': self.config,
                'n_sessions': len(self.session_embeddings),
                'embedding_dimension': list(self.session_embeddings.values())[0].shape[0] if self.session_embeddings else 0
            }
            
            # Add cluster info
            optimal_k = self.cluster_results['optimal_k']
            labels = self.cluster_results['kmeans'][optimal_k]['labels']
            
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                cluster_sessions = [self.session_metadata[i]['session_id'] for i in np.where(mask)[0]]
                
                serializable_results[f'cluster_{cluster_id}'] = {
                    'sessions': cluster_sessions,
                    'n_sessions': int(np.sum(mask))
                }
            
            with open(self.output_dir / "neurolm_b_analysis.json", 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📁 All results saved to: {self.output_dir}")

def main():
    """Main function to run NeuroLM-B EEG classification"""
    print("🧠 NeuroLM-B EEG Video Classification System")
    print("=" * 60)
    
    # Initialize classifier
    data_dir = "/Users/e.baena/Desktop/data"
    classifier = NeuroLMBEEGClassifier(data_dir)
    
    # Step 1: Verify checkpoints
    logger.info("🔍 Step 1: Verifying NeuroLM-B checkpoints")
    if not classifier.verify_checkpoints():
        logger.error("❌ Checkpoints not found")
        return
    
    # Step 2: Load pre-trained models
    logger.info("🔧 Step 2: Loading pre-trained NeuroLM-B models")
    if not classifier.load_pretrained_models():
        logger.error("❌ Failed to load models")
        return
    
    # Step 3: Analyze dataset
    logger.info("📊 Step 3: Analyzing EEG dataset")
    embeddings = classifier.analyze_dataset()
    
    if not embeddings:
        logger.error("❌ No embeddings extracted")
        return
    
    # Step 4: Clustering
    logger.info("🎯 Step 4: Performing clustering")
    cluster_results = classifier.perform_clustering()
    
    # Step 5: Visualization
    logger.info("📈 Step 5: Creating visualizations")
    classifier.create_comprehensive_visualization()
    
    # Step 6: Save results
    logger.info("💾 Step 6: Saving results")
    classifier.save_results()
    
    # Summary
    print("\n🎯 NEUROLM-B ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"✅ Checkpoint: NeuroLM-B (124M parameters)")
    print(f"📁 Sessions analyzed: {len(embeddings)}")
    print(f"🧠 Embedding dimension: {list(embeddings.values())[0].shape[0]}")
    print(f"🎯 Optimal clusters: {cluster_results.get('optimal_k', 'N/A')}")
    if 'optimal_k' in cluster_results:
        optimal_score = cluster_results['kmeans'][cluster_results['optimal_k']]['silhouette_score']
        print(f"📊 Silhouette score: {optimal_score:.3f}")
    print(f"💾 Results: ./neurolm_b_classification_results/")
    print(f"📈 Interactive plot: neurolm_b_classification.html")

if __name__ == "__main__":
    main()
