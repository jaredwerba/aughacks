#!/usr/bin/env python3
"""
Trigger Data Streaming for NeuroLM Web Dashboard
===============================================
Simple script to activate data streaming in the web dashboard system
"""

import asyncio
import websockets
import json
import time

async def trigger_streaming():
    """Connect to web dashboard and trigger data streaming"""
    try:
        # Connect to the web dashboard WebSocket
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            print("🔗 Connected to NeuroLM Web Dashboard")
            
            # Send start experiment command
            start_command = {
                "command": "start_experiment",
                "segment_duration": 10
            }
            
            await websocket.send(json.dumps(start_command))
            print("🎬 Sent start experiment command")
            
            # Wait for response
            response = await websocket.recv()
            print(f"📨 Response: {response}")
            
            # Keep connection alive for a bit to see streaming
            print("⏱️ Waiting for streaming data...")
            for i in range(10):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    if data.get('type') == 'metrics_update':
                        print(f"📊 Metrics: Attention={data.get('attention', 0):.3f}, "
                              f"Engagement={data.get('engagement', 0):.3f}")
                except asyncio.TimeoutError:
                    print(f"⏳ Waiting... ({i+1}/10)")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 Triggering NeuroLM Data Streaming...")
    asyncio.run(trigger_streaming())
