#!/usr/bin/env python3

import numpy as np
from pylsl import StreamInlet, resolve_streams
import time
import csv
from datetime import datetime
import os
import threading
import signal
import sys

class MultiStreamRecorder:
    def __init__(self):
        self.recording = False
        self.streams = []
        self.threads = []
        self.data_dir = "lsl_data"
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def find_all_streams(self):
        """Find all available LSL streams"""
        print("Looking for LSL streams...")
        streams = resolve_streams()
        
        if not streams:
            print("No LSL streams found. Make sure OpenBCI is streaming via LSL.")
            return []
        
        print(f"\nFound {len(streams)} stream(s):")
        stream_info = []
        
        for i, stream in enumerate(streams):
            info = {
                'index': i,
                'name': stream.name(),
                'type': stream.type(),
                'channel_count': stream.channel_count(),
                'nominal_srate': stream.nominal_srate(),
                'source_id': stream.source_id(),
                'stream_info': stream
            }
            stream_info.append(info)
            
            print(f"{i}: {info['name']}")
            print(f"   Type: {info['type']}")
            print(f"   Channels: {info['channel_count']}")
            print(f"   Rate: {info['nominal_srate']} Hz")
            print()
        
        return stream_info
    
    def record_single_stream(self, stream_info, duration):
        """Record data from a single stream"""
        stream_name = stream_info['name'].replace(' ', '_').replace('/', '_')
        filename = os.path.join(self.data_dir, f"{stream_name}_{self.session_timestamp}.csv")
        
        print(f"Recording {stream_info['name']} to {filename}")
        
        # Create inlet
        inlet = StreamInlet(stream_info['stream_info'])
        
        # Open CSV file
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write metadata
            csv_writer.writerow(['# LSL Stream Data'])
            csv_writer.writerow([f'# Session Start: {datetime.now().isoformat()}'])
            csv_writer.writerow([f'# Stream Name: {stream_info["name"]}'])
            csv_writer.writerow([f'# Stream Type: {stream_info["type"]}'])
            csv_writer.writerow([f'# Sampling Rate: {stream_info["nominal_srate"]} Hz'])
            csv_writer.writerow([f'# Channel Count: {stream_info["channel_count"]}'])
            csv_writer.writerow(['# Data Format: unix_timestamp, lsl_timestamp, system_time_iso, channel_1, channel_2, ...'])
            
            # Write headers
            headers = ['unix_timestamp', 'lsl_timestamp', 'system_time_iso']
            headers += [f'channel_{i+1}' for i in range(stream_info['channel_count'])]
            csv_writer.writerow(headers)
            
            # Record data
            start_time = time.time()
            sample_count = 0
            
            while self.recording and (duration == float('inf') or (time.time() - start_time) < duration):
                try:
                    sample, lsl_timestamp = inlet.pull_sample(timeout=0.1)
                    
                    if sample:
                        sample_count += 1
                        unix_timestamp = time.time()
                        system_time_iso = datetime.now().isoformat()
                        
                        # Write row
                        row = [unix_timestamp, lsl_timestamp, system_time_iso] + sample
                        csv_writer.writerow(row)
                        
                        # Print progress every 250 samples
                        if sample_count % 250 == 0:
                            print(f"  {stream_info['name']}: {sample_count} samples")
                
                except Exception as e:
                    print(f"Error recording {stream_info['name']}: {e}")
                    break
            
            elapsed = time.time() - start_time
            print(f"\n{stream_info['name']} recording complete:")
            print(f"  Total samples: {sample_count}")
            print(f"  Duration: {elapsed:.2f} seconds")
            print(f"  Effective rate: {sample_count/elapsed:.2f} Hz")
            print(f"  File: {filename}")
    
    def start_recording(self, stream_indices, duration=10):
        """Start recording from selected streams"""
        self.recording = True
        self.threads = []
        
        print(f"\nStarting recording for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        # Start a thread for each stream
        for idx in stream_indices:
            if idx < len(self.streams):
                stream = self.streams[idx]
                thread = threading.Thread(
                    target=self.record_single_stream,
                    args=(stream, duration)
                )
                thread.start()
                self.threads.append(thread)
        
        # Wait for threads to complete
        try:
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            print("\n\nStopping recording...")
            self.recording = False
            for thread in self.threads:
                thread.join()
    
    def stop_recording(self):
        """Stop all recordings"""
        self.recording = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nInterrupted by user')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("OpenBCI Multi-Stream LSL Recorder")
    print("=" * 50)
    
    recorder = MultiStreamRecorder()
    
    # Find all streams
    streams = recorder.find_all_streams()
    recorder.streams = streams
    
    if not streams:
        print("No streams available. Exiting.")
        return
    
    while True:
        print("\nOptions:")
        print("1. Record ALL streams (10 seconds)")
        print("2. Record ALL streams (custom duration)")
        print("3. Record ALL streams (continuous)")
        print("4. Select specific streams to record")
        print("5. Refresh stream list")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            all_indices = list(range(len(streams)))
            recorder.start_recording(all_indices, duration=10)
            
        elif choice == '2':
            duration = float(input("Enter duration in seconds: "))
            all_indices = list(range(len(streams)))
            recorder.start_recording(all_indices, duration=duration)
            
        elif choice == '3':
            all_indices = list(range(len(streams)))
            recorder.start_recording(all_indices, duration=float('inf'))
            
        elif choice == '4':
            print("\nAvailable streams:")
            for i, stream in enumerate(streams):
                print(f"{i}: {stream['name']} ({stream['type']})")
            
            selection = input("\nEnter stream numbers to record (comma-separated, e.g., 0,1,2): ")
            try:
                indices = [int(x.strip()) for x in selection.split(',')]
                duration = float(input("Duration in seconds (inf for continuous): "))
                recorder.start_recording(indices, duration=duration)
            except ValueError:
                print("Invalid input")
            
        elif choice == '5':
            streams = recorder.find_all_streams()
            recorder.streams = streams
            
        elif choice == '6':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("Error: pylsl library not installed")
        print("Install it with: pip install pylsl")
    except Exception as e:
        print(f"Error: {e}")