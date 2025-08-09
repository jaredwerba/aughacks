#!/usr/bin/env python3
"""
LSL-NeuroLM Real-Time EEG Attention System
Integrates Lab Streaming Layer (LSL) EEG data with NeuroLM for real-time attention/engagement metrics
Based on your LSL stream format: unix_timestamp,lsl_timestamp,system_time_iso,channel_1,...,channel_8
"""

import asyncio
import csv
import json
import logging
import numpy as np
import pandas as pd
import torch
import websockets
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import our NeuroLM components
from neurolm_tokenizer import NeuroLMTokenizer, NeuroTokenizerConfig
from neurolm_attention_model import NeuroLMAttentionModel, AttentionModelConfig
from neurolm_huggingface_integration import HuggingFaceNeuroLMLoader, HuggingFaceNeuroLMConfig

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Enable debug logging to see model outputs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSLNeuroLMProcessor:
    """Real-time LSL EEG processor with NeuroLM attention/engagement detection"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the LSL-NeuroLM processor"""
        self.config = config or self._default_config()
        
        # EEG data buffer for real-time processing
        self.eeg_buffer = deque(maxlen=self.config['buffer_size'])
        self.channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'C2']
        
        # NeuroLM components
        self.tokenizer = None
        self.attention_model = None
        self.hf_system = None
        
        # Real-time metrics
        self.current_attention = 0.0
        self.current_engagement = 0.0
        self.processing_stats = {
            'samples_processed': 0,
            'predictions_made': 0,
            'avg_processing_time': 0.0
        }
        
        # WebSocket clients for LLC transmission
        self.websocket_clients = set()
        
    def _default_config(self) -> Dict:
        """Default configuration for LSL-NeuroLM processing"""
        return {
            'sampling_rate': 250,           # Hz (from your LSL stream)
            'window_size': 200,             # samples (0.8 seconds at 250Hz)
            'overlap': 100,                 # samples (50% overlap)
            'buffer_size': 2000,            # samples (8 seconds buffer)
            'prediction_interval': 0.5,     # seconds between predictions
            'websocket_port': 8765,         # LLC transmission port
            'save_predictions': True,       # Save predictions to file
            'output_dir': './realtime_output'
        }
    
    async def initialize_neurolm(self) -> bool:
        """Initialize NeuroLM components"""
        try:
            logger.info("🔧 Initializing NeuroLM components...")
            
            # Tokenizer configuration
            tokenizer_config = NeuroTokenizerConfig(
                sampling_rate=self.config['sampling_rate'],
                window_size=self.config['window_size'],
                n_channels=8,
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
            
            # HuggingFace configuration
            hf_config = HuggingFaceNeuroLMConfig()
            
            # Initialize components
            self.tokenizer = NeuroLMTokenizer(tokenizer_config)
            self.attention_model = NeuroLMAttentionModel(attention_config)
            self.hf_system = HuggingFaceNeuroLMLoader(hf_config)
            
            logger.info("✅ NeuroLM components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize NeuroLM: {e}")
            return False
    
    def parse_lsl_line(self, line: str) -> Optional[Dict]:
        """Parse a single line from LSL CSV format"""
        try:
            parts = line.strip().split(',')
            if len(parts) < 11:  # unix_timestamp + lsl_timestamp + system_time + 8 channels
                return None
            
            # Extract timestamps and EEG data
            unix_timestamp = float(parts[0])
            lsl_timestamp = float(parts[1])
            system_time = parts[2]
            
            # Extract 8 EEG channels (skip any extra columns)
            eeg_channels = [float(parts[i]) for i in range(3, 11)]
            
            return {
                'unix_timestamp': unix_timestamp,
                'lsl_timestamp': lsl_timestamp,
                'system_time': system_time,
                'eeg_data': np.array(eeg_channels, dtype=np.float32),
                'channels': self.channel_names
            }
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse LSL line: {e}")
            return None
    
    def add_eeg_sample(self, sample: Dict) -> None:
        """Add EEG sample to processing buffer"""
        self.eeg_buffer.append(sample)
        self.processing_stats['samples_processed'] += 1
    
    def get_processing_window(self) -> Optional[np.ndarray]:
        """Get current EEG window for processing"""
        if len(self.eeg_buffer) < self.config['window_size']:
            return None
        
        # Extract last window_size samples
        recent_samples = list(self.eeg_buffer)[-self.config['window_size']:]
        eeg_matrix = np.array([sample['eeg_data'] for sample in recent_samples])
        
        # Transpose to (channels, samples) format for NeuroLM
        return eeg_matrix.T  # Shape: (8, window_size)
    
    async def process_realtime_attention(self) -> Optional[Dict]:
        """Process current EEG window for attention/engagement metrics"""
        try:
            # Get processing window
            eeg_window = self.get_processing_window()
            if eeg_window is None:
                return None
            
            start_time = datetime.now()
            
            # Convert to PyTorch tensor
            eeg_tensor = torch.tensor(eeg_window, dtype=torch.float32)
            
            # Tokenize EEG data
            tokenizer_output = self.tokenizer.forward(eeg_tensor, self.channel_names)
            
            # Extract tokens from tokenizer output (handle dict or tensor)
            if isinstance(tokenizer_output, dict):
                tokens = tokenizer_output.get('tokens', tokenizer_output.get('quantized', None))
                if tokens is None:
                    # Use the first tensor value in the dict
                    tokens = next(iter(tokenizer_output.values()))
            else:
                tokens = tokenizer_output
            
            # Ensure tokens is a tensor
            if not isinstance(tokens, torch.Tensor):
                logger.warning(f"Tokenizer output type: {type(tokens)}, converting to tensor")
                tokens = torch.tensor(tokens) if hasattr(tokens, '__iter__') else torch.zeros(1, 128)
            
            # Predict attention and engagement
            predictions = self.attention_model.forward(tokens)
            
            # Process HuggingFace integration (handle potential errors)
            try:
                hf_result = self.hf_system.process_eeg_tokens(tokens)
            except Exception as hf_error:
                logger.warning(f"HuggingFace integration failed: {hf_error}")
                hf_result = None
            
            # Extract metrics from NeuroLM model outputs
            logger.debug(f"Predictions type: {type(predictions)}")
            logger.debug(f"Predictions content: {predictions}")
            
            if isinstance(predictions, dict):
                # Get logits and convert to probabilities
                attention_logits = predictions.get('attention_logits', None)
                engagement_logits = predictions.get('engagement_logits', None)
                
                if attention_logits is not None:
                    # Convert logits to probabilities using softmax, then get confidence score
                    attention_probs = torch.softmax(attention_logits, dim=-1)
                    # Use max probability as attention score (0-1 range)
                    attention_score = float(torch.max(attention_probs, dim=-1)[0])
                else:
                    attention_score = 0.5
                
                if engagement_logits is not None:
                    # Convert logits to probabilities using softmax, then get confidence score
                    engagement_probs = torch.softmax(engagement_logits, dim=-1)
                    # Use max probability as engagement score (0-1 range)
                    engagement_score = float(torch.max(engagement_probs, dim=-1)[0])
                else:
                    engagement_score = 0.5
                
                # Also extract continuous metrics if available
                alpha_theta = predictions.get('alpha_theta_ratio', None)
                workload = predictions.get('workload', None)
                
                logger.debug(f"Attention logits shape: {attention_logits.shape if attention_logits is not None else 'None'}")
                logger.debug(f"Engagement logits shape: {engagement_logits.shape if engagement_logits is not None else 'None'}")
                logger.debug(f"Alpha/theta ratio: {alpha_theta}")
                logger.debug(f"Workload: {workload}")
                
            else:
                # If tensor output, use first two values
                attention_score = float(predictions[0]) if len(predictions) > 0 else 0.5
                engagement_score = float(predictions[1]) if len(predictions) > 1 else 0.5
            
            # Update current metrics
            self.current_attention = attention_score
            self.current_engagement = engagement_score
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * self.processing_stats['predictions_made'] + processing_time) /
                (self.processing_stats['predictions_made'] + 1)
            )
            self.processing_stats['predictions_made'] += 1
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'attention_score': attention_score,
                'engagement_score': engagement_score,
                'attention_level': self._classify_level(attention_score),
                'engagement_level': self._classify_level(engagement_score),
                'processing_time_ms': processing_time * 1000,
                'buffer_size': len(self.eeg_buffer),
                'hf_integration': bool(hf_result)
            }
            
            logger.info(f"📊 Attention: {attention_score:.3f} ({result['attention_level']}) | "
                       f"Engagement: {engagement_score:.3f} ({result['engagement_level']}) | "
                       f"Processing: {processing_time*1000:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Real-time processing failed: {e}")
            return None
    
    def _classify_level(self, score: float) -> str:
        """Classify attention/engagement score into levels"""
        if score < 0.33:
            return "Low"
        elif score < 0.67:
            return "Medium"
        else:
            return "High"
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for LLC transmission"""
        logger.info(f"🔗 New WebSocket client connected: {websocket.remote_address}")
        self.websocket_clients.add(websocket)
        
        try:
            await websocket.wait_closed()
        finally:
            self.websocket_clients.remove(websocket)
            logger.info(f"🔌 WebSocket client disconnected: {websocket.remote_address}")
    
    async def broadcast_metrics(self, metrics: Dict) -> None:
        """Broadcast metrics to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = json.dumps(metrics)
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    async def process_lsl_stream(self, lsl_file_path: str) -> None:
        """Process LSL stream file in real-time simulation"""
        logger.info(f"🎯 Starting LSL stream processing: {lsl_file_path}")
        
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Output file for predictions
        predictions_file = output_dir / f"neurolm_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            with open(lsl_file_path, 'r') as file:
                # Skip header lines
                for line in file:
                    if line.startswith('unix_timestamp'):
                        break
                
                prediction_counter = 0
                last_prediction_time = 0
                
                # Process each EEG sample
                for line in file:
                    sample = self.parse_lsl_line(line)
                    if sample is None:
                        continue
                    
                    # Add to buffer
                    self.add_eeg_sample(sample)
                    
                    # Check if it's time for a new prediction
                    current_time = sample['unix_timestamp']
                    if (current_time - last_prediction_time) >= self.config['prediction_interval']:
                        
                        # Process attention/engagement
                        metrics = await self.process_realtime_attention()
                        if metrics:
                            # Broadcast via WebSocket (LLC transmission)
                            await self.broadcast_metrics(metrics)
                            
                            # Save predictions
                            if self.config['save_predictions']:
                                self._save_prediction(predictions_file, metrics)
                            
                            prediction_counter += 1
                            last_prediction_time = current_time
                    
                    # Simulate real-time processing delay
                    await asyncio.sleep(0.001)  # 1ms delay
        
        except FileNotFoundError:
            logger.error(f"❌ LSL file not found: {lsl_file_path}")
        except Exception as e:
            logger.error(f"❌ LSL stream processing failed: {e}")
    
    def _save_prediction(self, file_path: Path, metrics: Dict) -> None:
        """Save prediction to CSV file"""
        file_exists = file_path.exists()
        
        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'attention_score', 'engagement_score', 
                         'attention_level', 'engagement_level', 'processing_time_ms', 
                         'buffer_size', 'hf_integration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(metrics)
    
    async def start_realtime_system(self, lsl_file_path: str) -> None:
        """Start the complete real-time LSL-NeuroLM system"""
        logger.info("🚀 Starting LSL-NeuroLM Real-Time System...")
        
        # Initialize NeuroLM
        if not await self.initialize_neurolm():
            logger.error("❌ Failed to initialize NeuroLM components")
            return
        
        # Start WebSocket server for LLC transmission
        websocket_server = await websockets.serve(
            self.websocket_handler, 
            "localhost", 
            self.config['websocket_port']
        )
        logger.info(f"🌐 WebSocket server started on port {self.config['websocket_port']}")
        
        # Start LSL stream processing
        await self.process_lsl_stream(lsl_file_path)
        
        # Print final statistics
        logger.info("📊 Final Processing Statistics:")
        logger.info(f"  Samples processed: {self.processing_stats['samples_processed']}")
        logger.info(f"  Predictions made: {self.processing_stats['predictions_made']}")
        logger.info(f"  Avg processing time: {self.processing_stats['avg_processing_time']*1000:.1f}ms")
        logger.info(f"  Final attention: {self.current_attention:.3f}")
        logger.info(f"  Final engagement: {self.current_engagement:.3f}")

async def main():
    """Main function to run the LSL-NeuroLM system"""
    # Configuration
    config = {
        'sampling_rate': 250,
        'window_size': 200,
        'overlap': 100,
        'buffer_size': 2000,
        'prediction_interval': 0.5,
        'websocket_port': 8765,
        'save_predictions': True,
        'output_dir': './realtime_output'
    }
    
    # LSL file path (use your desktop file)
    lsl_file_path = "/Users/e.baena/Desktop/lsl_stream_20250809_144838.csv"
    
    # Create and start the system
    processor = LSLNeuroLMProcessor(config)
    await processor.start_realtime_system(lsl_file_path)

if __name__ == "__main__":
    print("🧠 LSL-NeuroLM Real-Time EEG Attention System")
    print("=" * 50)
    asyncio.run(main())
