#!/usr/bin/env python3
"""
Integrated NeuroLM Real-Time Streaming System
============================================
Complete system that generates real-time EEG metrics and streams them to the dashboard
"""

import asyncio
import websockets
import json
import numpy as np
import time
import threading
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedStreamingSystem:
    """Integrated system for real-time EEG metrics streaming"""
    
    def __init__(self):
        self.connected_clients = set()
        self.streaming = False
        self.websocket_server = None
        self.segment_duration = 10  # Default 10 seconds
        self.eeg_source = 'simulator'  # Default source
        
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self.websocket_server = await websockets.serve(
                self.handle_client,
                'localhost',
                8765
            )
            logger.info("🌐 WebSocket server started on ws://localhost:8765")
            logger.info("🎬 Dashboard ready at: http://localhost:8080")
            
            # Start the metrics generation loop
            asyncio.create_task(self.generate_metrics_loop())
            
        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            raise
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connections"""
        client_addr = websocket.remote_address
        logger.info(f"🔗 Client connected: {client_addr}")
        
        self.connected_clients.add(websocket)
        
        try:
            # Send initial status
            await self.send_to_client(websocket, {
                'type': 'status',
                'message': 'Connected to Integrated Streaming System',
                'streaming_ready': True,
                'timestamp': time.time()
            })
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_command(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"❌ Client error: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_command(self, websocket, data):
        """Handle commands from dashboard"""
        command = data.get('command')
        
        if command == 'start_experiment':
            # Capture parameters from dashboard
            self.segment_duration = int(data.get('segment_duration', 10))
            self.eeg_source = data.get('eeg_source', 'simulator')
            has_video = data.get('has_video', False)
            
            logger.info(f"🎬 Starting streaming experiment - {self.eeg_source} mode, {self.segment_duration}s segments, Video: {has_video}")
            self.streaming = True
            await self.broadcast({
                'type': 'experiment_started',
                'message': f'Streaming started - {self.eeg_source} mode, {self.segment_duration}s segments',
                'segment_duration': self.segment_duration,
                'eeg_source': self.eeg_source,
                'has_video': has_video,
                'timestamp': time.time()
            })
            
        elif command == 'stop_experiment':
            logger.info("⏹️ Stopping streaming experiment")
            self.streaming = False
            await self.broadcast({
                'type': 'experiment_stopped',
                'message': 'Streaming stopped',
                'timestamp': time.time()
            })
            
        elif command == 'get_status':
            await self.send_to_client(websocket, {
                'type': 'status',
                'streaming': self.streaming,
                'clients_connected': len(self.connected_clients),
                'timestamp': time.time()
            })
    
    async def generate_metrics_loop(self):
        """Generate realistic EEG metrics continuously"""
        logger.info("🧠 Starting metrics generation loop")
        
        # Base values for realistic variation
        base_attention = 0.5
        base_engagement = 0.5
        base_workload = 0.3
        base_alpha_theta = 1.2
        
        while True:
            if self.streaming and self.connected_clients:
                # Generate realistic varying metrics
                attention = max(0.0, min(1.0, base_attention + np.random.normal(0, 0.05)))
                engagement = max(0.0, min(1.0, base_engagement + np.random.normal(0, 0.05)))
                workload = max(0.0, min(1.0, base_workload + np.random.normal(0, 0.03)))
                alpha_theta = max(0.5, min(2.0, base_alpha_theta + np.random.normal(0, 0.1)))
                
                # Generate simulated EEG channels (8 channels)
                eeg_channels = []
                for i in range(8):
                    # Simulate realistic EEG values (-100 to +100 microvolts)
                    channel_value = np.random.normal(0, 20) + np.sin(time.time() * (i + 1)) * 10
                    eeg_channels.append(channel_value)
                
                # Create metrics update message
                metrics = {
                    'type': 'metrics_update',
                    'attention': round(attention, 3),
                    'engagement': round(engagement, 3),
                    'workload': round(workload, 3),
                    'alpha_theta_ratio': round(alpha_theta, 3),
                    'attention_level': self.get_level(attention),
                    'engagement_level': self.get_level(engagement),
                    'workload_level': self.get_level(workload),
                    'eeg_channels': [round(ch, 2) for ch in eeg_channels],
                    'timestamp': time.time()
                }
                
                # Broadcast to all connected clients
                await self.broadcast(metrics)
                
                # Log metrics with segment info
                logger.info(f"📊 Streaming: Attention={attention:.3f}, "
                          f"Engagement={engagement:.3f}, Workload={workload:.3f} "
                          f"[{self.eeg_source} mode, {self.segment_duration}s segments, interval={update_interval:.1f}s]")
                
                # Slowly vary base values for realistic drift
                base_attention += np.random.normal(0, 0.001)
                base_engagement += np.random.normal(0, 0.001)
                base_workload += np.random.normal(0, 0.0005)
                
                # Keep base values in reasonable ranges
                base_attention = max(0.2, min(0.8, base_attention))
                base_engagement = max(0.2, min(0.8, base_engagement))
                base_workload = max(0.1, min(0.6, base_workload))
            
            # Use segment duration for metrics generation interval
            # For smooth visualization, update more frequently than segment duration
            update_interval = min(1.0, self.segment_duration / 5.0)  # At least 5 updates per segment
            await asyncio.sleep(update_interval)
    
    def get_level(self, value):
        """Convert numeric value to level string"""
        if value < 0.33:
            return 'Low'
        elif value < 0.67:
            return 'Medium'
        else:
            return 'High'
    
    async def send_to_client(self, websocket, data):
        """Send data to a specific client"""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def broadcast(self, data):
        """Broadcast data to all connected clients"""
        if not self.connected_clients:
            return
        
        # Send to all connected clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected

async def main():
    """Main function to start the integrated system"""
    print("🎬 Integrated NeuroLM Streaming System")
    print("=" * 50)
    print("🧠 Real-time EEG metrics generation")
    print("📊 Direct dashboard streaming")
    print("=" * 50)
    
    system = IntegratedStreamingSystem()
    
    try:
        await system.start_server()
        
        # Keep server running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down system...")
    except Exception as e:
        logger.error(f"❌ System error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
