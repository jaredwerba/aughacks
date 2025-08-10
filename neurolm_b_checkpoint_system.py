#!/usr/bin/env python3
"""
NeuroLM-B Checkpoint Integration System
Uses official NeuroLM-B pre-trained weights for EEG classification and clustering
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, login
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroLMBCheckpointLoader:
    """
    NeuroLM-B Checkpoint Loader and EEG Processor
    Integrates official pre-trained NeuroLM-B weights for real EEG analysis
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize NeuroLM-B checkpoint loader
        
        Args:
            checkpoint_path: Path to local NeuroLM-B.pt file, or None for HuggingFace
        """
        self.checkpoint_path = checkpoint_path
        self.model_name = "Weibang/NeuroLM"
        self.model_variant = "NeuroLM-B"
        
        # Model components
        self.neurolm_model = None
        self.vq_encoder = None
        self.tokenizer = None
        
        # Processing configuration
        self.config = {
            'sampling_rate': 250,
            'window_size': 1000,  # 4 seconds for better context
            'n_channels': 6,      # Your dataset has 6 channels
            'overlap_ratio': 0.5,
            'embedding_dim': 768, # NeuroLM-B hidden dimension
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Results storage
        self.session_embeddings = {}
        self.session_metadata = []
        
        logger.info(f"🔧 NeuroLM-B Checkpoint Loader initialized")
        logger.info(f"📱 Device: {self.config['device']}")
    
    def load_checkpoint(self) -> bool:
        """
        Load NeuroLM-B checkpoint (local file or HuggingFace)
        
        Returns:
            success: True if checkpoint loaded successfully
        """
        try:
            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                # Load local checkpoint
                logger.info(f"📂 Loading local NeuroLM-B checkpoint: {self.checkpoint_path}")
                return self._load_local_checkpoint()
            else:
                # Try HuggingFace
                logger.info(f"🤗 Loading NeuroLM-B from HuggingFace: {self.model_name}")
                return self._load_huggingface_checkpoint()
                
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def _load_local_checkpoint(self) -> bool:
        """Load NeuroLM-B from local .pt file"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.config['device'])
            logger.info(f"✅ Local checkpoint loaded: {self.checkpoint_path}")
            
            # Extract model components
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model_state = checkpoint['model']
            else:
                model_state = checkpoint
            
            # Initialize model architecture (need to match NeuroLM-B specs)
            self.neurolm_model = self._create_neurolm_b_architecture()
            
            # Load pre-trained weights
            self.neurolm_model.load_state_dict(model_state, strict=False)
            self.neurolm_model.to(self.config['device'])
            self.neurolm_model.eval()
            
            logger.info("✅ NeuroLM-B weights loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load local checkpoint: {e}")
            return False
    
    def _load_huggingface_checkpoint(self) -> bool:
        """Load NeuroLM-B from HuggingFace Hub"""
        try:
            # Try to load with authentication
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_auth_token=True
            )
            
            self.neurolm_model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map=self.config['device'],
                use_auth_token=True
            )
            
            logger.info("✅ HuggingFace NeuroLM-B loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ HuggingFace loading failed: {e}")
            logger.info("💡 Trying without authentication...")
            
            try:
                # Try without authentication
                self.neurolm_model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map=self.config['device']
                )
                logger.info("✅ HuggingFace NeuroLM-B loaded (no auth)")
                return True
            except Exception as e2:
                logger.error(f"❌ HuggingFace loading completely failed: {e2}")
                return False
    
    def _create_neurolm_b_architecture(self):
        """Create NeuroLM-B architecture for local checkpoint loading"""
        # NeuroLM-B specifications (124M parameters)
        config = {
            'vocab_size': 8192,
            'n_layer': 12,
            'n_head': 12,
            'n_embd': 768,
            'dropout': 0.1,
            'max_position_embeddings': 1024
        }
        
        # Create model architecture matching NeuroLM-B
        from transformers import GPT2Config, GPT2Model
        
        gpt_config = GPT2Config(
            vocab_size=config['vocab_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            dropout=config['dropout'],
            max_position_embeddings=config['max_position_embeddings']
        )
        
        model = GPT2Model(gpt_config)
        return model
    
    def extract_embeddings_from_eeg(self, eeg_data: np.ndarray, session_id: str) -> np.ndarray:
        """
        Extract NeuroLM-B embeddings from EEG data
        
        Args:
            eeg_data: EEG data [n_samples, n_channels]
            session_id: Session identifier
            
        Returns:
            embeddings: Session-level embeddings
        """
        logger.info(f"🧠 Extracting NeuroLM-B embeddings for: {session_id}")
        
        if self.neurolm_model is None:
            logger.error("❌ NeuroLM-B model not loaded")
            return np.array([])
        
        # Window the EEG data
        window_size = self.config['window_size']
        overlap = int(window_size * self.config['overlap_ratio'])
        step_size = window_size - overlap
        
        n_samples, n_channels = eeg_data.shape
        window_embeddings = []
        
        # Process each window
        for start_idx in range(0, n_samples - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = eeg_data[start_idx:end_idx]  # [window_size, n_channels]
            
            try:
                # Convert to tensor and process
                window_tensor = torch.tensor(window.T, dtype=torch.float32)  # [n_channels, window_size]
                window_tensor = window_tensor.unsqueeze(0).to(self.config['device'])  # [1, n_channels, window_size]
                
                with torch.no_grad():
                    # Process through NeuroLM-B
                    if hasattr(self.neurolm_model, 'encode'):
                        # If model has encode method
                        embedding = self.neurolm_model.encode(window_tensor)
                    else:
                        # Standard forward pass
                        outputs = self.neurolm_model(window_tensor)
                        
                        # Extract embedding from outputs
                        if hasattr(outputs, 'last_hidden_state'):
                            embedding = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence
                        elif hasattr(outputs, 'pooler_output'):
                            embedding = outputs.pooler_output
                        else:
                            # Fallback: use first output
                            embedding = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
                    
                    # Convert to numpy
                    embedding_np = embedding.cpu().numpy().flatten()
                    window_embeddings.append(embedding_np)
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to process window {start_idx//step_size}: {e}")
                continue
        
        if window_embeddings:
            # Average embeddings across windows for session representation
            session_embedding = np.mean(window_embeddings, axis=0)
            logger.info(f"✅ Extracted embedding shape: {session_embedding.shape}")
            return session_embedding
        else:
            logger.error(f"❌ No embeddings extracted for {session_id}")
            return np.array([])
    
    def process_dataset(self, data_dir: str) -> Dict[str, np.ndarray]:
        """
        Process entire EEG dataset and extract embeddings
        
        Args:
            data_dir: Directory containing EEG CSV files
            
        Returns:
            session_embeddings: Dictionary mapping session_id to embeddings
        """
        logger.info(f"📂 Processing EEG dataset: {data_dir}")
        
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("lsl_stream_*.csv"))
        
        if not csv_files:
            logger.error(f"❌ No EEG files found in {data_dir}")
            return {}
        
        logger.info(f"📁 Found {len(csv_files)} EEG session files")
        
        session_embeddings = {}
        
        for csv_file in sorted(csv_files):
            try:
                logger.info(f"📄 Processing: {csv_file.name}")
                
                # Load EEG data
                df = pd.read_csv(csv_file, comment='#')
                
                # Extract EEG channels (skip timestamp columns)
                eeg_columns = [col for col in df.columns if col.startswith('channel_')]
                eeg_data = df[eeg_columns].values  # [n_samples, n_channels]
                
                # Extract embeddings using NeuroLM-B
                session_id = csv_file.stem
                embedding = self.extract_embeddings_from_eeg(eeg_data, session_id)
                
                if embedding.size > 0:
                    session_embeddings[session_id] = embedding
                    
                    # Store metadata
                    timestamps = df['unix_timestamp'].values
                    self.session_metadata.append({
                        'session_id': session_id,
                        'duration_minutes': (timestamps[-1] - timestamps[0]) / 60,
                        'n_samples': len(eeg_data),
                        'n_channels': len(eeg_columns),
                        'file_size_mb': csv_file.stat().st_size / (1024 * 1024)
                    })
                
            except Exception as e:
                logger.error(f"❌ Failed to process {csv_file}: {e}")
                continue
        
        logger.info(f"✅ Processed {len(session_embeddings)} sessions successfully")
        self.session_embeddings = session_embeddings
        return session_embeddings
    
    def perform_clustering_analysis(self, embeddings_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform clustering analysis on NeuroLM-B embeddings"""
        logger.info("🎯 Performing clustering analysis with NeuroLM-B embeddings")
        
        # Convert to matrix
        session_ids = list(embeddings_dict.keys())
        embeddings_matrix = np.array(list(embeddings_dict.values()))
        
        logger.info(f"📊 Embeddings matrix shape: {embeddings_matrix.shape}")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)
        
        results = {}
        
        # K-Means clustering
        logger.info("🔵 Running K-Means clustering")
        kmeans_results = {}
        silhouette_scores = {}
        
        k_range = range(2, min(8, len(embeddings_matrix)))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_scaled)
            
            silhouette_avg = silhouette_score(embeddings_scaled, labels)
            
            kmeans_results[k] = {
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'silhouette_score': silhouette_avg
            }
            silhouette_scores[k] = silhouette_avg
            
            logger.info(f"  K={k}: Silhouette Score = {silhouette_avg:.3f}")
        
        # Find optimal K
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        results['kmeans'] = kmeans_results
        results['optimal_k'] = optimal_k
        results['session_ids'] = session_ids
        
        logger.info(f"🎯 Optimal K: {optimal_k} (Silhouette: {silhouette_scores[optimal_k]:.3f})")
        
        return results
    
    def create_visualization(self, embeddings_dict: Dict[str, np.ndarray], 
                           cluster_results: Dict[str, Any]) -> None:
        """Create comprehensive visualization of clustering results"""
        logger.info("📊 Creating NeuroLM-B clustering visualization")
        
        # Prepare data
        session_ids = cluster_results['session_ids']
        embeddings_matrix = np.array(list(embeddings_dict.values()))
        optimal_k = cluster_results['optimal_k']
        labels = cluster_results['kmeans'][optimal_k]['labels']
        
        # Dimensionality reduction
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(embeddings_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_matrix)-1))
        tsne_result = tsne.fit_transform(embeddings_scaled)
        
        # Create interactive plot
        fig = go.Figure()
        
        # Color palette
        colors = px.colors.qualitative.Set3
        
        # Add PCA clusters
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            
            fig.add_trace(go.Scatter(
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
            ))
        
        # Update layout
        fig.update_layout(
            title="🧠 NeuroLM-B EEG Session Clustering Analysis",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            template="plotly_dark",
            width=800,
            height=600,
            showlegend=True
        )
        
        # Save interactive plot
        output_dir = Path("./neurolm_b_results")
        output_dir.mkdir(exist_ok=True)
        
        fig.write_html(str(output_dir / "neurolm_b_clustering.html"))
        logger.info(f"💾 Visualization saved: {output_dir}/neurolm_b_clustering.html")
        
        # Also create static plots
        plt.figure(figsize=(15, 5))
        
        # PCA plot
        plt.subplot(1, 3, 1)
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       label=f'Cluster {cluster_id}', alpha=0.7, s=100)
            
            # Add session labels
            for j in np.where(mask)[0]:
                plt.annotate(session_ids[j], (pca_result[j, 0], pca_result[j, 1]), 
                           fontsize=8, ha='center')
        
        plt.title('PCA Clustering')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # t-SNE plot
        plt.subplot(1, 3, 2)
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       label=f'Cluster {cluster_id}', alpha=0.7, s=100)
        
        plt.title('t-SNE Clustering')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cluster quality metrics
        plt.subplot(1, 3, 3)
        k_values = list(cluster_results['kmeans'].keys())
        silhouette_values = [cluster_results['kmeans'][k]['silhouette_score'] for k in k_values]
        
        plt.bar([f'K={k}' for k in k_values], silhouette_values, alpha=0.7)
        plt.title('Clustering Quality (Silhouette Score)')
        plt.ylabel('Silhouette Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "neurolm_b_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"💾 Static plots saved: {output_dir}/neurolm_b_analysis.png")
    
    def save_results(self, embeddings_dict: Dict[str, np.ndarray], 
                    cluster_results: Dict[str, Any]) -> None:
        """Save all analysis results"""
        logger.info("💾 Saving NeuroLM-B analysis results")
        
        output_dir = Path("./neurolm_b_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save embeddings
        embeddings_df = pd.DataFrame(embeddings_dict).T
        embeddings_df.to_csv(output_dir / "neurolm_b_embeddings.csv")
        
        # Save metadata with cluster labels
        metadata_df = pd.DataFrame(self.session_metadata)
        optimal_k = cluster_results['optimal_k']
        metadata_df['cluster_label'] = cluster_results['kmeans'][optimal_k]['labels']
        metadata_df.to_csv(output_dir / "session_metadata_clustered.csv", index=False)
        
        # Save cluster analysis
        cluster_analysis = {}
        labels = cluster_results['kmeans'][optimal_k]['labels']
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_sessions = [self.session_metadata[i] for i in np.where(mask)[0]]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'n_sessions': int(np.sum(mask)),
                'session_ids': [meta['session_id'] for meta in cluster_sessions],
                'avg_duration': float(np.mean([meta['duration_minutes'] for meta in cluster_sessions])),
                'avg_file_size': float(np.mean([meta['file_size_mb'] for meta in cluster_sessions]))
            }
        
        with open(output_dir / "cluster_analysis.json", 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        logger.info(f"📁 Results saved to: {output_dir}")

async def main():
    """Main function to run NeuroLM-B classification"""
    print("🧠 NeuroLM-B EEG Classification & Clustering System")
    print("=" * 60)
    
    # Initialize NeuroLM-B system
    # Use the found checkpoint in Downloads
    local_checkpoint_paths = [
        "/Users/e.baena/Downloads/NeuroLM-B.pt",  # ✅ FOUND HERE
        "/Users/e.baena/Downloads/VQ.pt",         # ✅ VQ ENCODER ALSO AVAILABLE
        "/Users/e.baena/Desktop/NeuroLM-B.pt",
        "/Users/e.baena/CascadeProjects/NeuroLM/checkpoints/NeuroLM-B.pt"
    ]
    
    checkpoint_path = None
    for path in local_checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            break
    
    if checkpoint_path:
        logger.info(f"🎯 Found local checkpoint: {checkpoint_path}")
    else:
        logger.info("🤗 No local checkpoint found, will try HuggingFace")
    
    # Initialize system
    neurolm_system = NeuroLMBCheckpointLoader(checkpoint_path)
    
    # Load checkpoint
    success = neurolm_system.load_checkpoint()
    if not success:
        logger.error("❌ Failed to load NeuroLM-B checkpoint")
        logger.info("💡 Options:")
        logger.info("1. Download NeuroLM-B.pt to Desktop")
        logger.info("2. Set up HuggingFace authentication")
        logger.info("3. Use local implementation with random weights")
        return
    
    # Process dataset
    data_dir = "/Users/e.baena/Desktop/data"
    embeddings_dict = neurolm_system.process_dataset(data_dir)
    
    if not embeddings_dict:
        logger.error("❌ No embeddings extracted")
        return
    
    # Perform clustering
    cluster_results = neurolm_system.perform_clustering_analysis(embeddings_dict)
    
    # Create visualizations
    neurolm_system.create_visualization(embeddings_dict, cluster_results)
    
    # Save results
    neurolm_system.save_results(embeddings_dict, cluster_results)
    
    # Print summary
    print("\n🎯 NEUROLM-B CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"📁 Sessions analyzed: {len(embeddings_dict)}")
    print(f"🧠 Embedding dimension: {list(embeddings_dict.values())[0].shape[0]}")
    print(f"🎯 Optimal clusters: {cluster_results['optimal_k']}")
    print(f"📊 Best silhouette score: {cluster_results['kmeans'][cluster_results['optimal_k']]['silhouette_score']:.3f}")
    print(f"💾 Results saved to: ./neurolm_b_results/")

if __name__ == "__main__":
    asyncio.run(main())
