#!/usr/bin/env python3
"""
NeuroLM Video Experiment System
Generates embeddings for non-overlapping windows of 5, 10, and 30 seconds
Designed for subjects watching videos with 8-channel OpenBCI EEG
Works with CSV files when no live LSL stream is available
"""

import asyncio
import csv
import json
import logging
import numpy as np
import torch
import websockets
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import NeuroLM components
from neurolm_tokenizer import NeuroLMTokenizer, NeuroTokenizerConfig
from neurolm_attention_model import NeuroLMAttentionModel, AttentionModelConfig
from neurolm_huggingface_integration import HuggingFaceNeuroLMLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoExperimentProcessor:
    """NeuroLM processor for video experiments with fixed-duration embeddings"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the video experiment processor"""
        self.config = config or self._default_config()
        
        # EEG data storage
        self.eeg_data = []
        self.channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'C2']
        
        # Experiment segments (non-overlapping windows)
        self.experiment_segments = self.config['experiment_segments']  # [5, 10, 30] seconds
        self.current_segment_idx = 0
        
        # NeuroLM components
        self.tokenizer = None
        self.attention_model = None
        self.hf_system = None
        
        # Results storage
        self.embeddings_results = []
        
        # WebSocket clients for real-time updates
        self.websocket_clients = set()
        
        # Processing stats
        self.processing_stats = {
            'segments_processed': 0,
            'embeddings_generated': 0,
            'total_experiment_time': 0.0
        }
        
    def _default_config(self) -> Dict:
        """Default configuration for video experiments"""
        return {
            'sampling_rate': 250,  # OpenBCI Cyton sampling rate
            'experiment_segments': [5, 10, 30],  # Segment durations in seconds
            'websocket_port': 8765,
            'save_embeddings': True,
            'output_dir': './video_experiment_output',
            'experiment_name': 'video_attention_study'
        }
    
    def initialize_neurolm_components(self):
        """Initialize NeuroLM tokenizer and attention model"""
        try:
            # Initialize tokenizer
            tokenizer_config = NeuroTokenizerConfig(
                n_channels=8,
                sampling_rate=self.config['sampling_rate'],
                window_size=200,
                n_embed=1024,
                embed_dim=128
            )
            self.tokenizer = NeuroLMTokenizer(tokenizer_config)
            logger.info("✅ NeuroLM tokenizer initialized")
            
            # Initialize attention model
            model_config = AttentionModelConfig(
                vocab_size=1024,
                n_layer=12,
                n_head=8,
                n_embd=512
            )
            self.attention_model = NeuroLMAttentionModel(model_config)
            logger.info("✅ NeuroLM attention model initialized")
            
            # Initialize HuggingFace integration
            try:
                self.hf_system = HuggingFaceNeuroLMLoader()
                logger.info("✅ HuggingFace NeuroLM integration initialized")
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace integration failed: {e}")
                self.hf_system = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize NeuroLM components: {e}")
            raise
    
    def load_eeg_from_csv(self, csv_file_path: str) -> List[Dict]:
        """Load EEG data from CSV file"""
        eeg_data = []
        
        try:
            with open(csv_file_path, 'r') as f:
                csv_reader = csv.reader(f)
                
                # Skip header if present
                first_line = next(csv_reader, None)
                if first_line and not self._is_numeric_line(first_line):
                    logger.info("Skipping header line")
                else:
                    # Process first line if it's numeric
                    if first_line:
                        parsed_line = self.parse_lsl_line(','.join(first_line))
                        if parsed_line:
                            eeg_data.append(parsed_line)
                
                # Process remaining lines
                for line in csv_reader:
                    parsed_line = self.parse_lsl_line(','.join(line))
                    if parsed_line:
                        eeg_data.append(parsed_line)
            
            logger.info(f"✅ Loaded {len(eeg_data)} EEG samples from {csv_file_path}")
            return eeg_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load CSV file: {e}")
            return []
    
    def _is_numeric_line(self, line: List[str]) -> bool:
        """Check if a line contains numeric data"""
        try:
            # Try to convert first few elements to float
            for i in range(min(3, len(line))):
                float(line[i])
            return True
        except ValueError:
            return False
    
    def parse_lsl_line(self, line: str) -> Optional[Dict]:
        """Parse LSL CSV line with format: unix_timestamp,lsl_timestamp,system_time_iso,channel_1,...,channel_8"""
        try:
            parts = line.strip().split(',')
            if len(parts) < 11:  # 3 timestamps + 8 channels
                return None
            
            # Skip comment lines
            if parts[0].startswith('#') or parts[0].startswith('"'):
                return None
            
            unix_timestamp = float(parts[0])
            lsl_timestamp = float(parts[1])
            system_time_iso = parts[2]  # Keep as string (ISO format)
            
            # Extract 8 EEG channels (positions 3-10)
            eeg_channels = [float(parts[i]) for i in range(3, 11)]
            
            return {
                'unix_timestamp': unix_timestamp,
                'lsl_timestamp': lsl_timestamp,
                'system_time_iso': system_time_iso,
                'eeg_channels': eeg_channels
            }
            
        except (ValueError, IndexError) as e:
            # Only log if it's not a header/comment line
            if not line.startswith('#') and not line.startswith('"') and not line.startswith('unix_timestamp'):
                logger.warning(f"Failed to parse LSL line: {e}")
            return None
    
    def extract_segments(self, eeg_data: List[Dict]) -> List[Tuple[int, np.ndarray]]:
        """Extract non-overlapping segments of 5s, 10s, 30s from EEG data"""
        segments = []
        current_idx = 0
        segment_type_idx = 0
        
        while current_idx < len(eeg_data) and segment_type_idx < len(self.experiment_segments):
            segment_duration = self.experiment_segments[segment_type_idx]
            required_samples = segment_duration * self.config['sampling_rate']
            
            # Check if we have enough samples for this segment
            if current_idx + required_samples <= len(eeg_data):
                # Extract segment data
                segment_data = []
                for i in range(current_idx, current_idx + required_samples):
                    segment_data.append(eeg_data[i]['eeg_channels'])
                
                segment_array = np.array(segment_data)  # Shape: [samples, 8_channels]
                segments.append((segment_duration, segment_array))
                
                logger.info(f"📊 Extracted {segment_duration}s segment: {segment_array.shape}")
                
                # Move to next segment
                current_idx += required_samples
                segment_type_idx += 1
            else:
                logger.warning(f"⚠️ Not enough data for {segment_duration}s segment")
                break
        
        logger.info(f"✅ Extracted {len(segments)} segments total")
        return segments
    
    def generate_embedding(self, segment_duration: int, eeg_segment: np.ndarray) -> Dict:
        """Generate NeuroLM embedding for a complete EEG segment"""
        try:
            start_time = datetime.now()
            
            # Convert to PyTorch tensor
            eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32)
            logger.info(f"🧠 Processing {segment_duration}s segment: {eeg_tensor.shape}")
            
            # Split long segments into smaller windows compatible with tokenizer
            max_window_size = 200  # Maximum samples the tokenizer can handle
            if eeg_tensor.shape[0] > max_window_size:
                logger.info(f"📏 Segment too long ({eeg_tensor.shape[0]} samples), splitting into windows of {max_window_size}")
                embeddings_list = []
                attention_scores = []
                engagement_scores = []
                
                # Process in overlapping windows
                step_size = max_window_size // 2  # 50% overlap
                for start_idx in range(0, eeg_tensor.shape[0] - max_window_size + 1, step_size):
                    end_idx = start_idx + max_window_size
                    window_tensor = eeg_tensor[start_idx:end_idx]
                    
                    # Process this window
                    window_embedding = self._process_single_window(window_tensor)
                    if window_embedding:
                        embeddings_list.append(window_embedding['embedding_vector'])
                        attention_scores.append(window_embedding['attention_score'])
                        engagement_scores.append(window_embedding['engagement_score'])
                
                if not embeddings_list:
                    logger.error("❌ No valid windows processed")
                    return None
                
                # Average embeddings and scores
                embedding_vector = np.mean(embeddings_list, axis=0)
                attention_score = np.mean(attention_scores)
                engagement_score = np.mean(engagement_scores)
                alpha_theta_value = 0.0
                workload_value = 0.0
                
                logger.info(f"✅ Averaged {len(embeddings_list)} windows for {segment_duration}s segment")
                
            else:
                # Process single window directly
                window_result = self._process_single_window(eeg_tensor)
                if not window_result:
                    return None
                
                embedding_vector = window_result['embedding_vector']
                attention_score = window_result['attention_score']
                engagement_score = window_result['engagement_score']
                alpha_theta_value = window_result.get('alpha_theta_ratio', 0.0)
                workload_value = window_result.get('workload', 0.0)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create embedding result
            embedding_result = {
                'segment_duration': segment_duration,
                'segment_samples': int(eeg_segment.shape[0]),
                'embedding_vector': embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector,
                'embedding_dimension': len(embedding_vector),
                'attention_score': attention_score,
                'engagement_score': engagement_score,
                'alpha_theta_ratio': alpha_theta_value,
                'workload': workload_value,
                'processing_time_ms': processing_time * 1000,
                'timestamp': datetime.now().isoformat(),
                'eeg_shape': list(eeg_segment.shape),
                'num_windows': len(embeddings_list) if eeg_tensor.shape[0] > max_window_size else 1
            }
            
            logger.info(f"✅ Embedding generated for {segment_duration}s segment")
            logger.info(f"   📊 Embedding dim: {len(embedding_vector)}, Windows: {embedding_result['num_windows']}")
            logger.info(f"   🎯 Attention: {attention_score:.3f}, Engagement: {engagement_score:.3f}")
            logger.info(f"   📈 Alpha/Theta: {alpha_theta_value:.3f}, Workload: {workload_value:.3f}")
            logger.info(f"   ⏱️ Processing time: {processing_time*1000:.1f}ms")
            
            return embedding_result
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to generate embedding for {segment_duration}s segment: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _process_single_window(self, eeg_tensor: torch.Tensor) -> Optional[Dict]:
        """Process a single EEG window that fits within tokenizer limits"""
        try:
            # Reshape EEG tensor to expected format: [batch, channels, time]
            # Input: [time, channels] -> Output: [1, channels, time]
            if eeg_tensor.dim() == 2:
                eeg_tensor = eeg_tensor.transpose(0, 1).unsqueeze(0)  # [time, channels] -> [1, channels, time]
            elif eeg_tensor.dim() == 1:
                eeg_tensor = eeg_tensor.unsqueeze(0).unsqueeze(0)  # [time] -> [1, 1, time]
            
            logger.debug(f"🔄 Reshaped EEG tensor: {eeg_tensor.shape}")
            
            # Tokenize EEG segment
            tokenizer_output = self.tokenizer.forward(eeg_tensor, self.channel_names)
            
            # Extract tokens
            if isinstance(tokenizer_output, dict):
                tokens = tokenizer_output.get('tokens', tokenizer_output.get('quantized', None))
                if tokens is None:
                    tokens = next(iter(tokenizer_output.values()))
            else:
                tokens = tokenizer_output
            
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens) if hasattr(tokens, '__iter__') else torch.zeros(1, 128)
            
            # Generate embedding using attention model
            model_output = self.attention_model.forward(tokens)
            
            # Extract the main embedding (hidden_states)
            if isinstance(model_output, dict) and 'hidden_states' in model_output:
                hidden_states = model_output['hidden_states']
                # Use mean pooling to get fixed-size embedding
                embedding_vector = hidden_states.mean(dim=1).detach().numpy()  # [batch, embedding_dim]
                if embedding_vector.ndim > 1:
                    embedding_vector = embedding_vector[0]  # Take first batch
            else:
                # Fallback: use token mean pooling
                embedding_vector = tokens.mean(dim=1).detach().numpy()
                if embedding_vector.ndim > 1:
                    embedding_vector = embedding_vector[0]
            
            # Process attention/engagement predictions
            if isinstance(model_output, dict):
                attention_logits = model_output.get('attention_logits', None)
                engagement_logits = model_output.get('engagement_logits', None)
                alpha_theta_ratio = model_output.get('alpha_theta_ratio', None)
                workload = model_output.get('workload', None)
                
                if attention_logits is not None and attention_logits.numel() > 0:
                    attention_probs = torch.softmax(attention_logits, dim=-1)
                    max_values = torch.max(attention_probs, dim=-1)
                    if isinstance(max_values, tuple):
                        attention_score = float(max_values[0].item() if max_values[0].numel() > 0 else 0.5)
                    else:
                        attention_score = float(max_values.item() if max_values.numel() > 0 else 0.5)
                else:
                    attention_score = 0.5
                
                if engagement_logits is not None and engagement_logits.numel() > 0:
                    engagement_probs = torch.softmax(engagement_logits, dim=-1)
                    max_values = torch.max(engagement_probs, dim=-1)
                    if isinstance(max_values, tuple):
                        engagement_score = float(max_values[0].item() if max_values[0].numel() > 0 else 0.5)
                    else:
                        engagement_score = float(max_values.item() if max_values.numel() > 0 else 0.5)
                else:
                    engagement_score = 0.5
                
                # Extract continuous metrics safely
                if alpha_theta_ratio is not None and alpha_theta_ratio.numel() > 0:
                    alpha_theta_value = float(alpha_theta_ratio.flatten()[0].item())
                else:
                    alpha_theta_value = 0.0
                    
                if workload is not None and workload.numel() > 0:
                    workload_value = float(workload.flatten()[0].item())
                else:
                    workload_value = 0.0
            else:
                attention_score = 0.5
                engagement_score = 0.5
                alpha_theta_value = 0.0
                workload_value = 0.0
            
            return {
                'embedding_vector': embedding_vector,
                'attention_score': attention_score,
                'engagement_score': engagement_score,
                'alpha_theta_ratio': alpha_theta_value,
                'workload': workload_value
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to process single window: {e}")
            return None
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create embedding result
            embedding_result = {
                'segment_duration': segment_duration,
                'segment_samples': int(eeg_segment.shape[0]),
                'embedding_vector': embedding_vector.tolist(),
                'embedding_dimension': len(embedding_vector),
                'attention_score': attention_score,
                'engagement_score': engagement_score,
                'alpha_theta_ratio': alpha_theta_value,
                'workload': workload_value,
                'processing_time_ms': processing_time * 1000,
                'timestamp': datetime.now().isoformat(),
                'eeg_shape': list(eeg_segment.shape),
                'tokens_shape': list(tokens.shape),
                'num_tokens': int(tokens.shape[1]) if tokens.ndim > 1 else int(tokens.shape[0])
            }
            
            logger.info(f"✅ Embedding generated for {segment_duration}s segment")
            logger.info(f"   📊 Embedding dim: {len(embedding_vector)}, Tokens: {embedding_result['num_tokens']}")
            logger.info(f"   🎯 Attention: {attention_score:.3f}, Engagement: {engagement_score:.3f}")
            logger.info(f"   📈 Alpha/Theta: {alpha_theta_value:.3f}, Workload: {workload_value:.3f}")
            logger.info(f"   ⏱️ Processing time: {processing_time*1000:.1f}ms")
            
            return embedding_result
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to generate embedding for {segment_duration}s segment: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def save_embedding_result(self, embedding_result: Dict):
        """Save embedding result to file"""
        if not self.config['save_embeddings']:
            return
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save individual embedding
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        segment_duration = embedding_result['segment_duration']
        filename = f"{self.config['experiment_name']}_embedding_{segment_duration}s_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(embedding_result, f, indent=2)
        
        logger.info(f"💾 Embedding saved to: {filepath}")
    
    def save_all_embeddings_summary(self):
        """Save summary of all embeddings"""
        if not self.embeddings_results:
            return
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"{self.config['experiment_name']}_summary_{timestamp}.json"
        summary_filepath = output_dir / summary_filename
        
        summary = {
            'experiment_name': self.config['experiment_name'],
            'total_segments': len(self.embeddings_results),
            'segment_durations': [r['segment_duration'] for r in self.embeddings_results],
            'processing_stats': self.processing_stats,
            'embeddings': self.embeddings_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Experiment summary saved to: {summary_filepath}")
    
    async def broadcast_embedding(self, embedding_result: Dict):
        """Broadcast embedding result to WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = {
            'type': 'embedding_result',
            'data': embedding_result
        }
        
        # Broadcast to all connected clients
        disconnected_clients = set()
        for client in self.websocket_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.websocket_clients.add(websocket)
        logger.info(f"🔌 WebSocket client connected. Total clients: {len(self.websocket_clients)}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"🔌 WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        try:
            server = await websockets.serve(
                self.websocket_handler,
                "localhost",
                self.config['websocket_port']
            )
            logger.info(f"🌐 WebSocket server started on ws://localhost:{self.config['websocket_port']}")
            return server
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            return None
    
    async def process_video_experiment(self, csv_file_path: str):
        """Main function to process video experiment"""
        try:
            experiment_start_time = datetime.now()
            
            # Initialize NeuroLM components
            logger.info("🧠 Initializing NeuroLM components...")
            self.initialize_neurolm_components()
            
            # Start WebSocket server
            websocket_server = await self.start_websocket_server()
            
            # Load EEG data from CSV
            logger.info(f"📂 Loading EEG data from: {csv_file_path}")
            eeg_data = self.load_eeg_from_csv(csv_file_path)
            
            if not eeg_data:
                logger.error("❌ No EEG data loaded")
                return
            
            # Extract segments
            logger.info("✂️ Extracting experiment segments...")
            segments = self.extract_segments(eeg_data)
            
            if not segments:
                logger.error("❌ No segments extracted")
                return
            
            # Process each segment
            logger.info(f"🎬 Processing {len(segments)} video experiment segments...")
            
            for i, (segment_duration, eeg_segment) in enumerate(segments):
                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 Processing Segment {i+1}/{len(segments)}: {segment_duration}s")
                logger.info(f"{'='*60}")
                
                # Generate embedding
                embedding_result = self.generate_embedding(segment_duration, eeg_segment)
                
                if embedding_result:
                    # Store result
                    self.embeddings_results.append(embedding_result)
                    
                    # Save to file
                    self.save_embedding_result(embedding_result)
                    
                    # Broadcast via WebSocket
                    await self.broadcast_embedding(embedding_result)
                    
                    # Update stats
                    self.processing_stats['segments_processed'] += 1
                    self.processing_stats['embeddings_generated'] += 1
                
                # Small delay between segments
                await asyncio.sleep(0.1)
            
            # Final summary
            experiment_end_time = datetime.now()
            total_time = (experiment_end_time - experiment_start_time).total_seconds()
            self.processing_stats['total_experiment_time'] = total_time
            
            logger.info(f"\n{'='*60}")
            logger.info("🎉 VIDEO EXPERIMENT COMPLETED!")
            logger.info(f"{'='*60}")
            logger.info(f"📊 Segments processed: {self.processing_stats['segments_processed']}")
            logger.info(f"🧠 Embeddings generated: {self.processing_stats['embeddings_generated']}")
            logger.info(f"⏱️ Total experiment time: {total_time:.1f}s")
            
            # Save experiment summary
            self.save_all_embeddings_summary()
            
            # Keep WebSocket server running for a bit
            if websocket_server:
                logger.info("🌐 WebSocket server will remain active for 30 seconds...")
                await asyncio.sleep(30)
                websocket_server.close()
                await websocket_server.wait_closed()
                logger.info("🌐 WebSocket server closed")
            
        except Exception as e:
            logger.error(f"❌ Video experiment failed: {e}")
            raise

async def main():
    """Main function to run the video experiment system"""
    # Configuration for video experiment
    config = {
        'sampling_rate': 250,
        'experiment_segments': [5, 10, 30],  # Non-overlapping segments
        'websocket_port': 8765,
        'save_embeddings': True,
        'output_dir': './video_experiment_output',
        'experiment_name': 'neurolm_video_attention_study'
    }
    
    # CSV file path (use your test file)
    csv_file_path = "/Users/e.baena/Desktop/lsl_stream_20250809_144838.csv"
    
    # Create and start the video experiment system
    processor = VideoExperimentProcessor(config)
    await processor.process_video_experiment(csv_file_path)

if __name__ == "__main__":
    print("🎬 NeuroLM Video Experiment System")
    print("=" * 60)
    print("📊 Generating embeddings for 5s, 10s, 30s segments")
    print("🧠 Non-overlapping windows for video attention study")
    print("=" * 60)
    asyncio.run(main())
