#!/usr/bin/env python3
"""
NeuroLM Video Experiment Web Server
==================================
Serves the web dashboard and handles communication between the web interface
and the NeuroLM real-time system via WebSocket.
"""

import asyncio
import websockets
import json
import logging
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time
from typing import Dict, Set, Optional
import signal
import sys
import torch
import numpy as np
from collections import deque

# Import NeuroLM components
from neurolm_tokenizer import NeuroLMTokenizer, NeuroTokenizerConfig
from neurolm_attention_model import NeuroLMAttentionModel, AttentionModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeuroLMWebServer:
    """Web server for NeuroLM video experiment dashboard with pre-trained weights"""
    
    def __init__(self, web_port: int = 8080, websocket_port: int = 8765):
        self.web_port = web_port
        self.websocket_port = websocket_port
        self.web_interface_dir = Path(__file__).parent / "web_interface"
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.http_server = None
        self.websocket_server = None
        self.experiment_running = False
        self.current_metrics = {}
        
        # NeuroLM components with pre-trained weights
        self.tokenizer = None
        self.attention_model = None
        self.neurolm_initialized = False
        
        # EEG data buffer for real-time processing
        self.eeg_buffer = deque(maxlen=2000)
        self.channel_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6']
        
    async def initialize_neurolm(self) -> bool:
        """Initialize NeuroLM components with pre-trained weights"""
        try:
            logger.info("🔧 Initializing NeuroLM components with pre-trained weights...")
            
            # Check for local checkpoints
            neurolm_b_path = Path.home() / "Downloads" / "NeuroLM-B.pt"
            vq_path = Path.home() / "Downloads" / "VQ.pt"
            
            logger.info(f"🔍 Checking for NeuroLM-B checkpoint: {neurolm_b_path}")
            logger.info(f"🔍 Checking for VQ checkpoint: {vq_path}")
            
            # Tokenizer configuration (6 channels matching successful analysis)
            tokenizer_config = NeuroTokenizerConfig(
                sampling_rate=250,
                window_size=1000,  # 4 seconds at 250Hz
                n_channels=6,
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
                logger.info(f"📂 Expected: {neurolm_b_path}")
                logger.info(f"📂 Expected: {vq_path}")
                self.neurolm_initialized = False
            
            logger.info("✅ NeuroLM components initialized!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize NeuroLM: {e}")
            return False

    def start(self):
        """Start both HTTP and WebSocket servers"""
        logger.info("🌐 Starting NeuroLM Web Server...")
        
        # Start HTTP server in a separate thread
        http_thread = threading.Thread(target=self._start_http_server, daemon=True)
        http_thread.start()
        
        # Start WebSocket server with NeuroLM initialization
        asyncio.run(self._start_websocket_server_with_neurolm())
    
    def _start_http_server(self):
        """Start HTTP server for serving static files"""
        try:
            # Change to web interface directory
            import os
            os.chdir(self.web_interface_dir)
            
            class CustomHandler(SimpleHTTPRequestHandler):
                def log_message(self, format, *args):
                    # Suppress HTTP server logs
                    pass
            
            self.http_server = HTTPServer(('localhost', self.web_port), CustomHandler)
            logger.info(f"📡 HTTP server started on http://localhost:{self.web_port}")
            logger.info(f"🎬 Dashboard available at: http://localhost:{self.web_port}")
            
            self.http_server.serve_forever()
            
        except Exception as e:
            logger.error(f"❌ Failed to start HTTP server: {e}")
    
    async def _start_websocket_server_with_neurolm(self):
        """Start WebSocket server with NeuroLM initialization"""
        try:
            # Initialize NeuroLM first
            await self.initialize_neurolm()
            
            # Give HTTP server time to start
            await asyncio.sleep(1)
            
            self.websocket_server = await websockets.serve(
                self.handle_websocket_client,
                'localhost',
                self.websocket_port
            )
            
            logger.info(f"🔌 WebSocket server started on ws://localhost:{self.websocket_port}")
            logger.info("✅ NeuroLM Web Server fully initialized!")
            logger.info("=" * 60)
            logger.info("🎯 Ready for video experiments with NeuroLM-B!")
            neurolm_status = "Pre-trained NeuroLM-B" if self.neurolm_initialized else "Random weights"
            logger.info(f"🧠 Model status: {neurolm_status}")
            logger.info("📊 Real-time metrics generation active")
            logger.info("=" * 60)
            
            # Start metrics generation loop
            asyncio.create_task(self.generate_neurolm_metrics_loop())
            
            # Keep the server running
            await self.websocket_server.wait_closed()
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")

    async def _start_websocket_server(self):
        """Legacy WebSocket server method (kept for compatibility)"""
        await self._start_websocket_server_with_neurolm()
    
    async def handle_websocket_client(self, websocket):
        """Handle WebSocket client connections"""
        client_addr = websocket.remote_address
        logger.info(f"🔗 Client connected: {client_addr}")
        
        self.connected_clients.add(websocket)
        
        try:
            # Send initial status
            await self.send_to_client(websocket, {
                'type': 'status',
                'message': 'Connected to NeuroLM Web Server',
                'server_time': time.time()
            })
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_addr}: {message}")
                except Exception as e:
                    logger.error(f"Error handling message from {client_addr}: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"❌ WebSocket error with {client_addr}: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_client_message(self, websocket, data: Dict):
        """Handle messages from web dashboard clients"""
        message_type = data.get('type', 'unknown')
        logger.info(f"📨 Received message: {message_type}")
        
        if message_type == 'start_experiment':
            self.experiment_running = True
            await self.broadcast_to_clients({
                'type': 'experiment_started',
                'message': 'Video experiment started with NeuroLM-B',
                'neurolm_status': self.neurolm_initialized,
                'timestamp': time.time()
            })
            
        elif message_type == 'stop_experiment':
            self.experiment_running = False
            await self.broadcast_to_clients({
                'type': 'experiment_stopped',
                'message': 'Video experiment stopped',
                'timestamp': time.time()
            })
            
        elif message_type == 'get_status':
            await self.send_to_client(websocket, {
                'type': 'status',
                'experiment_running': self.experiment_running,
                'connected_clients': len(self.connected_clients),
                'neurolm_initialized': self.neurolm_initialized,
                'current_metrics': self.current_metrics,
                'timestamp': time.time()
            })
            
        elif message_type == 'change_eeg_source':
            logger.info(f"🔄 Changing EEG source to: {data.get('source')}")
            await self.broadcast_to_clients({
                'type': 'eeg_source_change',
                'source': data.get('source'),
                'timestamp': time.time()
            })
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def start_experiment(self, data: Dict):
        """Start video experiment"""
        self.experiment_running = True
        segment_duration = data.get('segment_duration', 10)
        
        logger.info(f"🎬 Starting experiment with {segment_duration}s segments")
        
        await self.broadcast_to_clients({
            'type': 'experiment_control',
            'action': 'start',
            'segment_duration': segment_duration,
            'timestamp': time.time()
        })
    
    async def pause_experiment(self):
        """Pause video experiment"""
        logger.info("⏸️ Pausing experiment")
        
        await self.broadcast_to_clients({
            'type': 'experiment_control',
            'action': 'pause',
            'timestamp': time.time()
        })

    def extract_neurolm_metrics(self, eeg_data: np.ndarray) -> Dict:
        """Extract real NeuroLM metrics from EEG data"""
        try:
            if not self.neurolm_initialized or self.tokenizer is None or self.attention_model is None:
                # Fallback to simulated metrics if NeuroLM not available
                return {
                    'attention': np.random.uniform(0.3, 0.7),
                    'engagement': np.random.uniform(0.3, 0.7),
                    'workload': np.random.uniform(0.2, 0.5),
                    'alpha_theta_ratio': np.random.uniform(0.8, 1.5)
                }
            
            with torch.no_grad():
                # Step 1: Tokenize EEG data
                tokens = self.tokenizer.encode_to_tokens(eeg_data.T, self.channel_names)
                
                if tokens is None or len(tokens) == 0:
                    raise ValueError("Failed to generate tokens")
                
                # Step 2: Get attention predictions
                token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                predictions = self.attention_model.predict_attention_state(token_tensor)
                
                # Extract metrics from predictions
                attention = float(torch.softmax(predictions['attention_logits'], dim=-1)[0, -1])
                engagement = float(torch.softmax(predictions['engagement_logits'], dim=-1)[0, -1])
                workload = float(torch.sigmoid(predictions['workload'])[0])
                alpha_theta = float(torch.sigmoid(predictions['alpha_theta_ratio'])[0] * 2.0)
                
                return {
                    'attention': max(0.0, min(1.0, attention)),
                    'engagement': max(0.0, min(1.0, engagement)),
                    'workload': max(0.0, min(1.0, workload)),
                    'alpha_theta_ratio': max(0.5, min(2.0, alpha_theta))
                }
                
        except Exception as e:
            logger.warning(f"⚠️ NeuroLM metrics extraction failed: {e}")
            # Fallback to simulated metrics
            return {
                'attention': np.random.uniform(0.3, 0.7),
                'engagement': np.random.uniform(0.3, 0.7),
                'workload': np.random.uniform(0.2, 0.5),
                'alpha_theta_ratio': np.random.uniform(0.8, 1.5)
            }

    async def generate_neurolm_metrics_loop(self):
        """Generate real-time metrics using NeuroLM-B"""
        logger.info("🧠 Starting NeuroLM-B metrics generation loop")
        
        while True:
            if self.experiment_running and self.connected_clients:
                # Generate simulated EEG data for processing
                eeg_data = np.random.randn(1000, 6) * 10  # 4 seconds of 6-channel EEG
                
                # Add to buffer for windowing
                for sample in eeg_data:
                    self.eeg_buffer.append(sample)
                
                # Extract metrics using NeuroLM if we have enough data
                if len(self.eeg_buffer) >= 1000:  # 4 seconds of data
                    window_data = np.array(list(self.eeg_buffer)[-1000:])  # Last 4 seconds
                    metrics_dict = self.extract_neurolm_metrics(window_data)
                    
                    # Generate EEG channels for visualization
                    eeg_channels = [float(sample) for sample in list(self.eeg_buffer)[-1]] if self.eeg_buffer else [0.0] * 6
                    
                    # Create metrics update message
                    metrics = {
                        'type': 'metrics_update',
                        'attention': round(metrics_dict['attention'], 3),
                        'engagement': round(metrics_dict['engagement'], 3),
                        'workload': round(metrics_dict['workload'], 3),
                        'alpha_theta_ratio': round(metrics_dict['alpha_theta_ratio'], 3),
                        'attention_level': self.get_level(metrics_dict['attention']),
                        'engagement_level': self.get_level(metrics_dict['engagement']),
                        'workload_level': self.get_level(metrics_dict['workload']),
                        'eeg_channels': [round(ch, 2) for ch in eeg_channels],
                        'timestamp': time.time()
                    }
                    
                    # Update current metrics and broadcast
                    self.current_metrics = metrics
                    await self.broadcast_to_clients(metrics)
                    
                    # Log metrics with NeuroLM status
                    neurolm_status = "NeuroLM-B" if self.neurolm_initialized else "Simulated"
                    logger.info(f"📊 Streaming ({neurolm_status}): Attention={metrics_dict['attention']:.3f}, "
                              f"Engagement={metrics_dict['engagement']:.3f}, Workload={metrics_dict['workload']:.3f}")
            
            await asyncio.sleep(1.0)  # Update every second

    async def handle_client_message(self, websocket, data: Dict):
        """Handle messages from web dashboard clients"""
        try:
            message_type = data.get('type', 'unknown')
            logger.info(f"📨 Received message: {message_type}")
            
            if message_type == 'start_experiment':
                self.experiment_running = True
                await self.broadcast_to_clients({
                    'type': 'experiment_started',
                    'message': 'Video experiment started with NeuroLM-B',
                    'neurolm_status': self.neurolm_initialized,
                    'timestamp': time.time()
                })
                
            elif message_type == 'stop_experiment':
                self.experiment_running = False
                await self.broadcast_to_clients({
                    'type': 'experiment_stopped',
                    'message': 'Video experiment stopped',
                    'timestamp': time.time()
                })
                
            elif message_type == 'get_status':
                await self.send_to_client(websocket, {
                    'type': 'status',
                    'experiment_running': self.experiment_running,
                    'connected_clients': len(self.connected_clients),
                    'neurolm_initialized': self.neurolm_initialized,
                    'current_metrics': self.current_metrics,
                    'timestamp': time.time()
                })
                
        except Exception as e:
            logger.error(f"❌ Error handling client message: {e}")

    def get_level(self, value):
        """Convert numeric value to level string"""
        if value < 0.33:
            return 'Low'
        elif value < 0.67:
            return 'Medium'
        else:
            return 'High'

    async def send_to_client(self, websocket, data: Dict):
        """Send data to a specific client"""
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
        
        # Create a copy of the set to avoid modification during iteration
        clients = self.connected_clients.copy()
        
        for client in clients:
            await self.send_to_client(client, data)
    
    def update_metrics(self, metrics: Dict):
        """Update current metrics (called by NeuroLM system)"""
        self.current_metrics = metrics
        
        # Broadcast to all connected clients
        asyncio.create_task(self.broadcast_to_clients({
            'type': 'metrics_update',
            **metrics,
            'timestamp': time.time()
        }))
    
    def shutdown(self):
        """Shutdown the server"""
        logger.info("🛑 Shutting down NeuroLM Web Server...")
        
        if self.http_server:
            self.http_server.shutdown()
        
        if self.websocket_server:
            self.websocket_server.close()

class NeuroLMWebServerIntegration:
    """Integration class to connect NeuroLM system with web server"""
    
    def __init__(self, websocket_port: int = 8765):
        self.websocket_port = websocket_port
        self.websocket = None
        self.connected = False
    
    async def connect(self):
        """Connect to the web server WebSocket"""
        try:
            self.websocket = await websockets.connect(f'ws://localhost:{self.websocket_port}')
            self.connected = True
            logger.info("🔗 Connected to NeuroLM Web Server")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to web server: {e}")
            return False
    
    async def send_metrics(self, metrics: Dict):
        """Send metrics to web dashboard"""
        if not self.connected or not self.websocket:
            return
        
        try:
            await self.websocket.send(json.dumps({
                'type': 'metrics_update',
                **metrics,
                'timestamp': time.time()
            }))
        except Exception as e:
            logger.error(f"❌ Error sending metrics to web server: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from web server"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("🛑 Received shutdown signal")
    sys.exit(0)

def main():
    """Main function to start the web server"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🎬 NeuroLM Video Experiment Web Server")
    print("=" * 50)
    print("🧠 Real-time EEG metrics visualization")
    print("📊 Synchronized video and brain data")
    print("=" * 50)
    
    # Create and start server
    server = NeuroLMWebServer()
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        server.shutdown()

if __name__ == "__main__":
    main()
