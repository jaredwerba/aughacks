#!/usr/bin/env python3
"""
NeuroLM Embedding Extractor for Video Database
Extracts embeddings from EEG data during video experiments and stores them in vector database
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json

# Import NeuroLM components
from neurolm_tokenizer import NeuroLMTokenizer, NeuroTokenizerConfig
from neurolm_attention_model import NeuroLMAttentionModel, AttentionModelConfig
from vector_database import VideoEmbeddingVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuroLMEmbeddingExtractor:
    """Extract and store NeuroLM embeddings for video experiments"""
    
    def __init__(self, db_path: str = "video_embeddings.db"):
        self.tokenizer = None
        self.attention_model = None
        self.neurolm_initialized = False
        self.vector_db = VideoEmbeddingVectorDB(db_path)
        
        # EEG processing parameters
        self.sampling_rate = 250
        self.window_size = 1000  # 4 seconds
        self.n_channels = 6
        self.channel_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6']
        
    async def initialize_neurolm(self) -> bool:
        """Initialize NeuroLM components with pre-trained weights"""
        try:
            logger.info("🔧 Initializing NeuroLM components for embedding extraction...")
            
            # Check for local checkpoints
            neurolm_b_path = Path.home() / "Downloads" / "NeuroLM-B.pt"
            vq_path = Path.home() / "Downloads" / "VQ.pt"
            
            logger.info(f"🔍 Checking for NeuroLM-B checkpoint: {neurolm_b_path}")
            logger.info(f"🔍 Checking for VQ checkpoint: {vq_path}")
            
            # Tokenizer configuration
            tokenizer_config = NeuroTokenizerConfig(
                sampling_rate=self.sampling_rate,
                window_size=self.window_size,
                n_channels=self.n_channels,
                n_embed=8192,
                embed_dim=128
            )
            
            # Attention model configuration
            attention_config = AttentionModelConfig(
                vocab_size=8192,
                n_layer=12,
                n_head=12,
                n_embd=768,
                dropout=0.1,
                n_attention_classes=3,
                n_engagement_classes=3
            )
            
            # Initialize components
            self.tokenizer = NeuroLMTokenizer(tokenizer_config)
            self.attention_model = NeuroLMAttentionModel(attention_config)
            
            # Load pre-trained weights if available
            if neurolm_b_path.exists() and vq_path.exists():
                logger.info("🎯 Loading pre-trained NeuroLM-B weights...")
                
                try:
                    # Load VQ encoder weights
                    vq_checkpoint = torch.load(vq_path, map_location='cpu', weights_only=False)
                    if hasattr(self.tokenizer, 'quantizer') and isinstance(vq_checkpoint, dict):
                        if 'embedding.weight' in vq_checkpoint:
                            self.tokenizer.quantizer.embedding.weight.data = vq_checkpoint['embedding.weight']
                            logger.info("✅ VQ encoder weights loaded")
                    
                    # Load NeuroLM-B model weights
                    neurolm_checkpoint = torch.load(neurolm_b_path, map_location='cpu', weights_only=False)
                    if isinstance(neurolm_checkpoint, dict):
                        # Load compatible weights into attention model
                        model_state = self.attention_model.state_dict()
                        loaded_weights = 0
                        
                        for key, value in neurolm_checkpoint.items():
                            if key in model_state and model_state[key].shape == value.shape:
                                model_state[key] = value
                                loaded_weights += 1
                        
                        self.attention_model.load_state_dict(model_state, strict=False)
                        logger.info(f"✅ NeuroLM-B weights loaded ({loaded_weights} layers)")
                    
                    logger.info("🎯 Pre-trained models loaded successfully!")
                    self.neurolm_initialized = True
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load pre-trained weights: {e}")
                    logger.info("🔄 Using randomly initialized weights")
                    self.neurolm_initialized = False
            else:
                logger.warning("⚠️ Pre-trained checkpoints not found, using random initialization")
                self.neurolm_initialized = False
            
            logger.info("✅ NeuroLM embedding extractor initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize NeuroLM: {e}")
            return False
    
    def extract_embeddings_from_eeg(self, eeg_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract NeuroLM embeddings from EEG data"""
        try:
            if not self.neurolm_initialized or self.tokenizer is None or self.attention_model is None:
                logger.warning("⚠️ NeuroLM not initialized, cannot extract embeddings")
                return None
            
            # Ensure EEG data is the right shape (samples, channels)
            if eeg_data.shape[1] != self.n_channels:
                logger.error(f"❌ EEG data has {eeg_data.shape[1]} channels, expected {self.n_channels}")
                return None
            
            if eeg_data.shape[0] < self.window_size:
                logger.warning(f"⚠️ EEG data too short: {eeg_data.shape[0]} samples, need {self.window_size}")
                return None
            
            with torch.no_grad():
                # Step 1: Tokenize EEG data
                tokens = self.tokenizer.encode_to_tokens(eeg_data.T, self.channel_names)
                
                if tokens is None or len(tokens) == 0:
                    logger.error("❌ Failed to generate tokens from EEG data")
                    return None
                
                # Step 2: Extract embeddings from attention model
                token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                
                # Get hidden states (embeddings) from the model
                embeddings = self.attention_model.get_embeddings(token_tensor)
                
                if embeddings is None:
                    logger.error("❌ Failed to extract embeddings from model")
                    return None
                
                # Average embeddings across sequence length to get fixed-size representation
                if embeddings.dim() == 3:  # (batch, seq_len, embed_dim)
                    embeddings = torch.mean(embeddings, dim=1)  # (batch, embed_dim)
                
                # Convert to numpy
                embeddings_np = embeddings.cpu().numpy()
                
                logger.info(f"✅ Extracted embeddings: shape {embeddings_np.shape}")
                return embeddings_np
                
        except Exception as e:
            logger.error(f"❌ Failed to extract embeddings: {e}")
            return None
    
    def process_video_experiment(self, 
                               video_id: str,
                               eeg_data: np.ndarray,
                               video_url: str = None,
                               video_title: str = None,
                               experiment_metrics: Dict = None,
                               duration_seconds: float = None) -> bool:
        """Process a complete video experiment and store embeddings"""
        try:
            logger.info(f"🎬 Processing video experiment: {video_id}")
            
            # Extract embeddings from EEG data
            embeddings = self.extract_embeddings_from_eeg(eeg_data)
            
            if embeddings is None:
                logger.error(f"❌ Failed to extract embeddings for video: {video_id}")
                return False
            
            # Prepare experiment date
            experiment_date = datetime.now().isoformat()
            
            # Store in vector database
            success = self.vector_db.store_video_embedding(
                video_id=video_id,
                embedding=embeddings,
                video_url=video_url,
                video_title=video_title,
                eeg_metrics=experiment_metrics,
                experiment_date=experiment_date,
                duration_seconds=duration_seconds
            )
            
            if success:
                logger.info(f"✅ Successfully stored embeddings for video: {video_id}")
            else:
                logger.error(f"❌ Failed to store embeddings for video: {video_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to process video experiment: {e}")
            return False
    
    def search_similar_videos(self, 
                            eeg_data: np.ndarray, 
                            k: int = 5) -> List[Dict]:
        """Search for videos with similar EEG patterns"""
        try:
            logger.info(f"🔍 Searching for {k} similar videos based on EEG patterns")
            
            # Extract embeddings from query EEG data
            query_embeddings = self.extract_embeddings_from_eeg(eeg_data)
            
            if query_embeddings is None:
                logger.error("❌ Failed to extract query embeddings")
                return []
            
            # Search in vector database
            results = self.vector_db.search_similar_videos(
                query_embedding=query_embeddings,
                k=k,
                include_metadata=True
            )
            
            logger.info(f"✅ Found {len(results)} similar videos")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search similar videos: {e}")
            return []
    
    def batch_process_eeg_segments(self, 
                                 video_id: str,
                                 eeg_data: np.ndarray,
                                 segment_duration: float = 30.0,
                                 video_url: str = None,
                                 video_title: str = None) -> List[Dict]:
        """Process EEG data in segments and extract embeddings for each"""
        try:
            logger.info(f"📊 Batch processing EEG data for video: {video_id}")
            
            segment_samples = int(segment_duration * self.sampling_rate)
            total_samples = eeg_data.shape[0]
            
            results = []
            segment_count = 0
            
            for start_idx in range(0, total_samples - self.window_size, segment_samples):
                end_idx = min(start_idx + segment_samples, total_samples)
                
                if end_idx - start_idx < self.window_size:
                    break  # Skip segments that are too short
                
                segment_data = eeg_data[start_idx:end_idx]
                segment_id = f"{video_id}_segment_{segment_count:03d}"
                
                # Extract embeddings for this segment
                embeddings = self.extract_embeddings_from_eeg(segment_data)
                
                if embeddings is not None:
                    # Calculate segment metrics
                    segment_metrics = {
                        'segment_id': segment_id,
                        'start_time': start_idx / self.sampling_rate,
                        'end_time': end_idx / self.sampling_rate,
                        'duration': (end_idx - start_idx) / self.sampling_rate,
                        'embedding_norm': float(np.linalg.norm(embeddings)),
                        'embedding_mean': float(np.mean(embeddings)),
                        'embedding_std': float(np.std(embeddings))
                    }
                    
                    # Store segment embedding
                    success = self.vector_db.store_video_embedding(
                        video_id=segment_id,
                        embedding=embeddings,
                        video_url=video_url,
                        video_title=f"{video_title} (Segment {segment_count})" if video_title else f"Segment {segment_count}",
                        eeg_metrics=segment_metrics,
                        experiment_date=datetime.now().isoformat(),
                        duration_seconds=segment_metrics['duration']
                    )
                    
                    if success:
                        results.append(segment_metrics)
                        logger.info(f"✅ Processed segment {segment_count}: {segment_metrics['duration']:.1f}s")
                    else:
                        logger.warning(f"⚠️ Failed to store segment {segment_count}")
                
                segment_count += 1
            
            logger.info(f"✅ Batch processing complete: {len(results)} segments processed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to batch process EEG segments: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get vector database statistics"""
        return self.vector_db.get_database_stats()
    
    def list_videos(self, limit: int = 50) -> List[Dict]:
        """List videos in the database"""
        return self.vector_db.list_all_videos(limit)


async def main():
    """Test the embedding extractor"""
    print("🧠 NeuroLM Embedding Extractor")
    print("=" * 50)
    
    # Initialize extractor
    extractor = NeuroLMEmbeddingExtractor()
    
    # Initialize NeuroLM
    success = await extractor.initialize_neurolm()
    if not success:
        print("❌ Failed to initialize NeuroLM")
        return
    
    # Get database stats
    stats = extractor.get_database_stats()
    print(f"📊 Database Stats: {stats}")
    
    # Generate sample EEG data for testing
    sample_eeg = np.random.randn(2000, 6) * 10  # 8 seconds of 6-channel EEG
    
    # Test embedding extraction
    embeddings = extractor.extract_embeddings_from_eeg(sample_eeg)
    if embeddings is not None:
        print(f"✅ Test embedding extraction successful: {embeddings.shape}")
        
        # Test storing video experiment
        success = extractor.process_video_experiment(
            video_id="test_video_001",
            eeg_data=sample_eeg,
            video_url="https://youtube.com/watch?v=test",
            video_title="Test Video",
            experiment_metrics={'attention': 0.75, 'engagement': 0.65},
            duration_seconds=8.0
        )
        
        if success:
            print("✅ Test video experiment stored successfully")
            
            # Test similarity search
            similar_videos = extractor.search_similar_videos(sample_eeg, k=3)
            print(f"🔍 Found {len(similar_videos)} similar videos")
            for video in similar_videos:
                print(f"  - {video['video_id']}: similarity={video['similarity_score']:.3f}")
    else:
        print("❌ Test embedding extraction failed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
