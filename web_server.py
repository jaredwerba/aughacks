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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NeuroLMWebServer:
    """Web server for NeuroLM video experiment dashboard"""
    
    def __init__(self, web_port: int = 8080, websocket_port: int = 8765):
        self.web_port = web_port
        self.websocket_port = websocket_port
        self.web_interface_dir = Path(__file__).parent / "web_interface"
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.http_server = None
        self.websocket_server = None
        self.experiment_running = False
        self.current_metrics = {}
        
    def start(self):
        """Start both HTTP and WebSocket servers"""
        logger.info("🌐 Starting NeuroLM Web Server...")
        
        # Start HTTP server in a separate thread
        http_thread = threading.Thread(target=self._start_http_server, daemon=True)
        http_thread.start()
        
        # Start WebSocket server
        asyncio.run(self._start_websocket_server())
    
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
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time communication"""
        try:
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
            logger.info("🎯 Ready for video experiments!")
            logger.info("📊 Connect your NeuroLM system to start streaming metrics")
            logger.info("=" * 60)
            
            # Keep the server running
            await self.websocket_server.wait_closed()
            
        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket server: {e}")
    
    async def handle_websocket_client(self, websocket, path):
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
        command = data.get('command')
        
        if command == 'start_experiment':
            await self.start_experiment(data)
        elif command == 'pause_experiment':
            await self.pause_experiment()
        elif command == 'stop_experiment':
            await self.stop_experiment()
        elif command == 'change_eeg_source':
            await self.change_eeg_source(data.get('source'))
        elif command == 'get_status':
            await self.send_status(websocket)
        else:
            logger.warning(f"Unknown command: {command}")
    
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
    
    async def stop_experiment(self):
        """Stop video experiment"""
        self.experiment_running = False
        
        logger.info("⏹️ Stopping experiment")
        
        await self.broadcast_to_clients({
            'type': 'experiment_control',
            'action': 'stop',
            'timestamp': time.time()
        })
    
    async def change_eeg_source(self, source: str):
        """Change EEG data source"""
        logger.info(f"🔄 Changing EEG source to: {source}")
        
        await self.broadcast_to_clients({
            'type': 'eeg_source_change',
            'source': source,
            'timestamp': time.time()
        })
    
    async def send_status(self, websocket):
        """Send current status to client"""
        status = {
            'type': 'status',
            'experiment_running': self.experiment_running,
            'connected_clients': len(self.connected_clients),
            'current_metrics': self.current_metrics,
            'timestamp': time.time()
        }
        
        await self.send_to_client(websocket, status)
    
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
