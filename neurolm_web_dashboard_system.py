#!/usr/bin/env python3
"""
NeuroLM Web Dashboard Integration System
=======================================
Integrates NeuroLM real-time processing with web dashboard visualization
Combines video experiment functionality with real-time web interface
"""

import asyncio
import websockets
import json
import logging
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import time

# Import NeuroLM components
from neurolm_tokenizer import NeuroLMTokenizer, NeuroTokenizerConfig
from neurolm_attention_model import NeuroLMAttentionModel, AttentionModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeuroLMWebDashboardSystem:
    """Integrated NeuroLM system with web dashboard visualization"""
    
    def __init__(self, websocket_port: int = 8765):
        self.websocket_port = websocket_port
        self.connected_clients = set()
        self.websocket_server = None
        
        # NeuroLM components
        self.tokenizer = None
        self.attention_model = None
        self.channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
        self.sampling_rate = 250
        self.window_size = 200  # 0.8 seconds at 250Hz
        
        # Experiment state
        self.experiment_running = False
        self.current_segment_duration = 10
        self.eeg_data = None
        self.current_metrics = {}
        
        # Data buffers for real-time processing
        self.eeg_buffer = []
        self.metrics_history = []
        
    async def initialize(self):
        """Initialize NeuroLM components and web server"""
        logger.info("🧠 Initializing NeuroLM Web Dashboard System...")
        
        # Initialize NeuroLM components
        await self._initialize_neurolm()
        
        # Start WebSocket server
        await self._start_websocket_server()
        
        logger.info("✅ NeuroLM Web Dashboard System ready!")
    
    async def _initialize_neurolm(self):
        """Initialize NeuroLM tokenizer and attention model"""
        try:
            # Initialize tokenizer
            tokenizer_config = NeuroTokenizerConfig(
                n_channels=8,
                sampling_rate=self.sampling_rate,
                window_size=self.window_size,
                n_embed=8192,
                embed_dim=128,
                n_layer=6,
                n_head=8,
                n_embd=512
            )
            
            self.tokenizer = NeuroLMTokenizer(tokenizer_config)
            logger.info("✅ NeuroLM tokenizer initialized")
            
            # Initialize attention model
            model_config = AttentionModelConfig(
                vocab_size=8192,  # Use tokenizer's n_embed as vocab_size
                n_embd=512,
                n_head=8,
                n_layer=6,
                n_attention_classes=3,
                n_engagement_classes=3,
                use_auxiliary_tasks=True
            )
            
            self.attention_model = NeuroLMAttentionModel(model_config)
            logger.info("✅ NeuroLM attention model initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing NeuroLM: {e}")
            raise
    
    async def _start_websocket_server(self):
        """Start WebSocket server for web dashboard communication"""
        try:
            self.websocket_server = await websockets.serve(
                self.handle_websocket_client,
                'localhost',
                self.websocket_port
            )
            
            logger.info(f"🌐 WebSocket server started on ws://localhost:{self.websocket_port}")
            logger.info("🎬 Web dashboard ready at: http://localhost:8080")
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
            raise
    
    async def handle_websocket_client(self, websocket):
        """Handle WebSocket client connections from web dashboard"""
        client_addr = websocket.remote_address
        logger.info(f"🔗 Dashboard client connected: {client_addr}")
        
        self.connected_clients.add(websocket)
        
        try:
            # Send initial status
            await self.send_to_client(websocket, {
                'type': 'status',
                'message': 'Connected to NeuroLM System',
                'neurolm_ready': True,
                'timestamp': time.time()
            })
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_dashboard_command(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from dashboard: {message}")
                except Exception as e:
                    logger.error(f"Error handling dashboard message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Dashboard client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_dashboard_command(self, websocket, data: Dict):
        """Handle commands from web dashboard"""
        command = data.get('command')
        
        if command == 'start_experiment':
            await self.start_video_experiment(data)
        elif command == 'pause_experiment':
            await self.pause_experiment()
        elif command == 'stop_experiment':
            await self.stop_experiment()
        elif command == 'load_eeg_data':
            await self.load_eeg_data(data.get('file_path'))
        elif command == 'get_status':
            await self.send_status(websocket)
        else:
            logger.warning(f"Unknown dashboard command: {command}")
    
    async def start_video_experiment(self, data: Dict):
        """Start video experiment with specified parameters"""
        self.experiment_running = True
        self.current_segment_duration = int(data.get('segment_duration', 10))
        
        logger.info(f"🎬 Starting video experiment ({self.current_segment_duration}s segments)")
        
        # Broadcast to dashboard
        await self.broadcast_to_clients({
            'type': 'experiment_started',
            'segment_duration': self.current_segment_duration,
            'timestamp': time.time()
        })
        
        # Start processing if EEG data is available
        if self.eeg_data is not None:
            asyncio.create_task(self.process_video_experiment())
    
    async def pause_experiment(self):
        """Pause video experiment"""
        logger.info("⏸️ Pausing video experiment")
        
        await self.broadcast_to_clients({
            'type': 'experiment_paused',
            'timestamp': time.time()
        })
    
    async def stop_experiment(self):
        """Stop video experiment"""
        self.experiment_running = False
        
        logger.info("⏹️ Stopping video experiment")
        
        await self.broadcast_to_clients({
            'type': 'experiment_stopped',
            'timestamp': time.time()
        })
    
    async def load_eeg_data(self, file_path: str):
        """Load EEG data from CSV file"""
        try:
            logger.info(f"📂 Loading EEG data from: {file_path}")
            
            # Load CSV data (reuse logic from video experiment system)
            valid_lines = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line or 'timestamp' in line.lower():
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 11:
                        eeg_values = parts[-8:]
                        try:
                            eeg_floats = [float(val) for val in eeg_values]
                            valid_lines.append(eeg_floats)
                        except ValueError:
                            continue
            
            if valid_lines:
                self.eeg_data = np.array(valid_lines)
                logger.info(f"✅ Loaded {self.eeg_data.shape[0]} EEG samples")
                
                await self.broadcast_to_clients({
                    'type': 'eeg_data_loaded',
                    'samples': self.eeg_data.shape[0],
                    'channels': self.eeg_data.shape[1],
                    'duration_seconds': self.eeg_data.shape[0] / self.sampling_rate,
                    'timestamp': time.time()
                })
            else:
                raise ValueError("No valid EEG data found")
                
        except Exception as e:
            logger.error(f"❌ Error loading EEG data: {e}")
            await self.broadcast_to_clients({
                'type': 'error',
                'message': f"Failed to load EEG data: {e}",
                'timestamp': time.time()
            })
    
    async def process_video_experiment(self):
        """Process video experiment with real-time metrics"""
        if self.eeg_data is None:
            logger.warning("No EEG data available for processing")
            return
        
        logger.info("🎬 Starting video experiment processing...")
        
        # Calculate segment parameters
        segment_samples = self.current_segment_duration * self.sampling_rate
        total_samples = self.eeg_data.shape[0]
        
        segment_count = 0
        current_position = 0
        
        while self.experiment_running and current_position < total_samples:
            # Extract segment
            end_position = min(current_position + segment_samples, total_samples)
            segment_data = self.eeg_data[current_position:end_position]
            
            if segment_data.shape[0] < segment_samples:
                logger.info("⚠️ Not enough data for full segment, stopping")
                break
            
            segment_count += 1
            logger.info(f"🎯 Processing segment {segment_count} ({self.current_segment_duration}s)")
            
            # Process segment and generate metrics
            start_time = time.time()
            metrics = await self.process_eeg_segment(segment_data)
            processing_time = (time.time() - start_time) * 1000
            
            if metrics:
                metrics['processing_time_ms'] = processing_time
                metrics['segment_number'] = segment_count
                metrics['segment_duration'] = self.current_segment_duration
                
                # Store in history
                self.metrics_history.append(metrics)
                self.current_metrics = metrics
                
                # Broadcast to dashboard
                await self.broadcast_to_clients({
                    'type': 'metrics_update',
                    **metrics,
                    'timestamp': time.time()
                })
                
                logger.info(f"📊 Segment {segment_count}: Attention={metrics['attention']:.3f}, "
                          f"Engagement={metrics['engagement']:.3f}, Processing={processing_time:.1f}ms")
            
            # Move to next segment
            current_position = end_position
            
            # Simulate real-time by waiting for segment duration
            await asyncio.sleep(self.current_segment_duration)
        
        logger.info("🎬 Video experiment processing completed")
        await self.broadcast_to_clients({
            'type': 'experiment_completed',
            'total_segments': segment_count,
            'timestamp': time.time()
        })
    
    async def process_eeg_segment(self, segment_data: np.ndarray) -> Optional[Dict]:
        """Process EEG segment and generate NeuroLM metrics"""
        try:
            # Convert to tensor
            eeg_tensor = torch.tensor(segment_data, dtype=torch.float32)
            
            # Split long segments into windows if necessary
            if eeg_tensor.shape[0] > self.window_size:
                windows = self._split_segment_into_windows(eeg_tensor)
                window_results = []
                
                for window in windows:
                    result = await self._process_single_window(window)
                    if result:
                        window_results.append(result)
                
                if window_results:
                    # Average results across windows
                    return self._average_window_results(window_results)
            else:
                # Process single window
                return await self._process_single_window(eeg_tensor)
                
        except Exception as e:
            logger.error(f"❌ Error processing EEG segment: {e}")
            return None
    
    def _split_segment_into_windows(self, eeg_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split long EEG segment into smaller windows"""
        windows = []
        step_size = self.window_size // 2  # 50% overlap
        
        for i in range(0, eeg_tensor.shape[0] - self.window_size + 1, step_size):
            window = eeg_tensor[i:i + self.window_size]
            windows.append(window)
        
        return windows
    
    async def _process_single_window(self, eeg_tensor: torch.Tensor) -> Optional[Dict]:
        """Process single EEG window"""
        try:
            # Reshape tensor: [time, channels] -> [batch, channels, time]
            if eeg_tensor.dim() == 2:
                eeg_tensor = eeg_tensor.transpose(0, 1).unsqueeze(0)
            
            # Tokenize
            tokenizer_output = self.tokenizer.forward(eeg_tensor, self.channel_names)
            token_ids = tokenizer_output['tokens']
            
            # Run through attention model
            with torch.no_grad():
                model_output = self.attention_model.forward(token_ids)
            
            # Extract metrics
            attention_logits = model_output['attention_logits']
            engagement_logits = model_output['engagement_logits']
            
            attention_probs = torch.softmax(attention_logits, dim=-1)
            engagement_probs = torch.softmax(engagement_logits, dim=-1)
            
            # Calculate scores (weighted average)
            attention_score = float(torch.sum(attention_probs * torch.tensor([0.0, 0.5, 1.0])))
            engagement_score = float(torch.sum(engagement_probs * torch.tensor([0.0, 0.5, 1.0])))
            
            # Extract other metrics
            alpha_theta_ratio = float(model_output.get('alpha_theta_ratio', torch.tensor([0.0])).squeeze())
            workload = float(model_output.get('workload', torch.tensor([0.0])).squeeze())
            
            # Generate embedding
            hidden_states = model_output['hidden_states']
            embedding = torch.mean(hidden_states, dim=1).squeeze().tolist()
            
            return {
                'attention': attention_score,
                'engagement': engagement_score,
                'alpha_theta_ratio': alpha_theta_ratio,
                'workload': workload,
                'embedding': embedding,
                'embedding_dimension': len(embedding),
                'attention_level': self._get_level(attention_score),
                'engagement_level': self._get_level(engagement_score),
                'eeg_channels': eeg_tensor.squeeze().mean(dim=0).tolist()  # Average EEG values
            }
            
        except Exception as e:
            logger.error(f"❌ Error in single window processing: {e}")
            return None
    
    def _average_window_results(self, results: List[Dict]) -> Dict:
        """Average results from multiple windows"""
        if not results:
            return {}
        
        # Average numerical metrics
        avg_result = {
            'attention': np.mean([r['attention'] for r in results]),
            'engagement': np.mean([r['engagement'] for r in results]),
            'alpha_theta_ratio': np.mean([r['alpha_theta_ratio'] for r in results]),
            'workload': np.mean([r['workload'] for r in results]),
            'num_windows': len(results)
        }
        
        # Average embeddings
        embeddings = [r['embedding'] for r in results]
        avg_result['embedding'] = np.mean(embeddings, axis=0).tolist()
        avg_result['embedding_dimension'] = len(avg_result['embedding'])
        
        # Set levels based on averaged scores
        avg_result['attention_level'] = self._get_level(avg_result['attention'])
        avg_result['engagement_level'] = self._get_level(avg_result['engagement'])
        
        # Average EEG channels
        eeg_channels = [r['eeg_channels'] for r in results]
        avg_result['eeg_channels'] = np.mean(eeg_channels, axis=0).tolist()
        
        return avg_result
    
    def _get_level(self, score: float) -> str:
        """Convert score to level string"""
        if score < 0.33:
            return 'Low'
        elif score < 0.67:
            return 'Medium'
        else:
            return 'High'
    
    async def send_to_client(self, websocket, data: Dict):
        """Send data to specific client"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            self.connected_clients.discard(websocket)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def broadcast_to_clients(self, data: Dict):
        """Broadcast data to all connected clients"""
        if not self.connected_clients:
            return
        
        clients = self.connected_clients.copy()
        for client in clients:
            await self.send_to_client(client, data)
    
    async def send_status(self, websocket):
        """Send current status to client"""
        status = {
            'type': 'status',
            'experiment_running': self.experiment_running,
            'connected_clients': len(self.connected_clients),
            'eeg_data_loaded': self.eeg_data is not None,
            'current_metrics': self.current_metrics,
            'segment_duration': self.current_segment_duration,
            'timestamp': time.time()
        }
        
        await self.send_to_client(websocket, status)
    
    async def run_server(self):
        """Run the server indefinitely"""
        try:
            await self.websocket_server.wait_closed()
        except KeyboardInterrupt:
            logger.info("🛑 Server stopped by user")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")

async def main():
    """Main function to start the integrated system"""
    print("🎬 NeuroLM Web Dashboard System")
    print("=" * 50)
    print("🧠 Real-time EEG processing with web visualization")
    print("📊 Video experiment dashboard integration")
    print("=" * 50)
    
    # Create and initialize system
    system = NeuroLMWebDashboardSystem()
    
    try:
        await system.initialize()
        
        # Load default EEG data if available
        default_eeg_file = "/Users/e.baena/Desktop/lsl_stream_20250809_144838.csv"
        if Path(default_eeg_file).exists():
            await system.load_eeg_data(default_eeg_file)
        
        # Run server
        await system.run_server()
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
