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

class AnalysisStreamsRecorder:
    def __init__(self):
        self.recording = False
        self.threads = []
        self.data_dir = "lsl_data"
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Target stream types we want to record
        self.target_streams = {
            'timeseriesfilt': None,
            'bandpower': None,
            'fft': None
        }
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def find_target_streams(self):
        """Find the specific streams we want: TimeSeriesFilt, BandPower, FFT"""
        print("Looking for analysis streams (TimeSeriesFilt, BandPower, FFT)...")
        streams = resolve_streams()
        
        if not streams:
            print("No LSL streams found. Make sure OpenBCI is streaming via LSL.")
            return False
        
        print(f"\nFound {len(streams)} total stream(s)")
        print("Identifying target streams...\n")
        
        # Reset target streams
        for key in self.target_streams:
            self.target_streams[key] = None
        
        # Identify each stream
        for stream in streams:
            name_lower = stream.name().lower()
            type_lower = stream.type().lower()
            
            # Check for TimeSeriesFilt
            if 'timeseries' in name_lower or 'filt' in name_lower or 'filtered' in name_lower:
                self.target_streams['timeseriesfilt'] = {
                    'name': stream.name(),
                    'type': stream.type(),
                    'channel_count': stream.channel_count(),
                    'nominal_srate': stream.nominal_srate(),
                    'stream_info': stream
                }
                print(f"✓ TimeSeriesFilt found: {stream.name()}")
                print(f"  Channels: {stream.channel_count()}, Rate: {stream.nominal_srate()} Hz")
            
            # Check for BandPower
            elif 'band' in name_lower or 'power' in name_lower or 'bandpower' in name_lower:
                self.target_streams['bandpower'] = {
                    'name': stream.name(),
                    'type': stream.type(),
                    'channel_count': stream.channel_count(),
                    'nominal_srate': stream.nominal_srate(),
                    'stream_info': stream
                }
                print(f"✓ BandPower found: {stream.name()}")
                print(f"  Channels: {stream.channel_count()}, Rate: {stream.nominal_srate()} Hz")
            
            # Check for FFT
            elif 'fft' in name_lower or 'frequency' in name_lower or 'spectrum' in name_lower:
                self.target_streams['fft'] = {
                    'name': stream.name(),
                    'type': stream.type(),
                    'channel_count': stream.channel_count(),
                    'nominal_srate': stream.nominal_srate(),
                    'stream_info': stream
                }
                print(f"✓ FFT found: {stream.name()}")
                print(f"  Channels: {stream.channel_count()}, Rate: {stream.nominal_srate()} Hz")
            
            # Also check if it's just labeled as EEG but might be filtered
            elif 'eeg' in type_lower and self.target_streams['timeseriesfilt'] is None:
                self.target_streams['timeseriesfilt'] = {
                    'name': stream.name(),
                    'type': stream.type(),
                    'channel_count': stream.channel_count(),
                    'nominal_srate': stream.nominal_srate(),
                    'stream_info': stream
                }
                print(f"✓ EEG/TimeSeries found: {stream.name()}")
                print(f"  Channels: {stream.channel_count()}, Rate: {stream.nominal_srate()} Hz")
        
        # Report missing streams
        print("\n" + "="*50)
        missing = []
        found = []
        for stream_type, info in self.target_streams.items():
            if info is None:
                missing.append(stream_type)
            else:
                found.append(stream_type)
        
        if found:
            print(f"Ready to record: {', '.join(found)}")
        if missing:
            print(f"Not found: {', '.join(missing)}")
            print("\nMake sure these widgets are enabled in OpenBCI GUI:")
            if 'timeseriesfilt' in missing:
                print("  - Time Series widget (with filtering enabled)")
            if 'bandpower' in missing:
                print("  - Band Power widget")
            if 'fft' in missing:
                print("  - FFT widget")
        
        return len(found) > 0
    
    def record_stream(self, stream_type, stream_info, duration):
        """Record data from a single stream"""
        # Create filename based on stream type
        filename = os.path.join(self.data_dir, f"{stream_type}_{self.session_timestamp}.csv")
        
        print(f"\nRecording {stream_type.upper()}: {stream_info['name']}")
        print(f"  → {filename}")
        
        # Create inlet
        inlet = StreamInlet(stream_info['stream_info'])
        
        # Open CSV file
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write metadata
            csv_writer.writerow(['# OpenBCI Analysis Stream Data'])
            csv_writer.writerow([f'# Stream Type: {stream_type.upper()}'])
            csv_writer.writerow([f'# Session Start: {datetime.now().isoformat()}'])
            csv_writer.writerow([f'# Stream Name: {stream_info["name"]}'])
            csv_writer.writerow([f'# LSL Type: {stream_info["type"]}'])
            csv_writer.writerow([f'# Sampling Rate: {stream_info["nominal_srate"]} Hz'])
            csv_writer.writerow([f'# Channel Count: {stream_info["channel_count"]}'])
            
            # Add stream-specific metadata
            if stream_type == 'fft':
                csv_writer.writerow(['# Note: FFT bins represent frequency components'])
            elif stream_type == 'bandpower':
                csv_writer.writerow(['# Note: Each channel represents a frequency band (Delta, Theta, Alpha, Beta, Gamma)'])
            elif stream_type == 'timeseriesfilt':
                csv_writer.writerow(['# Note: Filtered time series EEG data'])
            
            csv_writer.writerow(['# Data Format: unix_timestamp, lsl_timestamp, system_time_iso, channel_1, channel_2, ...'])
            
            # Write headers
            headers = ['unix_timestamp', 'lsl_timestamp', 'system_time_iso']
            
            # Add appropriate channel headers based on stream type
            if stream_type == 'bandpower' and stream_info['channel_count'] == 5:
                headers += ['delta', 'theta', 'alpha', 'beta', 'gamma']
            elif stream_type == 'fft':
                headers += [f'bin_{i+1}' for i in range(stream_info['channel_count'])]
            else:
                headers += [f'channel_{i+1}' for i in range(stream_info['channel_count'])]
            
            csv_writer.writerow(headers)
            
            # Record data
            start_time = time.time()
            sample_count = 0
            last_print_time = start_time
            
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
                        
                        # Print progress every 2 seconds
                        if time.time() - last_print_time > 2:
                            print(f"  {stream_type.upper()}: {sample_count} samples recorded")
                            last_print_time = time.time()
                
                except Exception as e:
                    print(f"Error recording {stream_type}: {e}")
                    break
            
            elapsed = time.time() - start_time
            print(f"\n{stream_type.upper()} complete:")
            print(f"  Total samples: {sample_count}")
            print(f"  Duration: {elapsed:.2f} seconds")
            if elapsed > 0:
                print(f"  Effective rate: {sample_count/elapsed:.2f} Hz")
            print(f"  Saved to: {filename}")
    
    def start_recording(self, duration=10):
        """Start recording from all available target streams"""
        self.recording = True
        self.threads = []
        
        # Count available streams
        available_streams = [(k, v) for k, v in self.target_streams.items() if v is not None]
        
        if not available_streams:
            print("\nNo target streams available to record!")
            return
        
        print(f"\n{'='*50}")
        print(f"Starting {len(available_streams)}-stream recording")
        print(f"Duration: {duration if duration != float('inf') else 'Continuous'} seconds")
        print(f"Session ID: {self.session_timestamp}")
        print("Press Ctrl+C to stop")
        print('='*50)
        
        # Start a thread for each available stream
        for stream_type, stream_info in available_streams:
            thread = threading.Thread(
                target=self.record_stream,
                args=(stream_type, stream_info, duration)
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
        
        print(f"\n{'='*50}")
        print("Recording session complete!")
        print(f"All files saved to: {self.data_dir}/")
        print('='*50)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nInterrupted by user')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("OpenBCI Analysis Streams Recorder")
    print("Target streams: TimeSeriesFilt, BandPower, FFT")
    print("=" * 50)
    
    recorder = AnalysisStreamsRecorder()
    
    # Find target streams
    if not recorder.find_target_streams():
        print("\nNo target streams found. Please check OpenBCI GUI configuration.")
        return
    
    while True:
        print("\n" + "="*50)
        print("Recording Options:")
        print("1. Quick record (10 seconds)")
        print("2. Standard record (30 seconds)")
        print("3. Extended record (60 seconds)")
        print("4. Custom duration")
        print("5. Continuous recording")
        print("6. Refresh stream list")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            recorder.start_recording(duration=10)
            
        elif choice == '2':
            recorder.start_recording(duration=30)
            
        elif choice == '3':
            recorder.start_recording(duration=60)
            
        elif choice == '4':
            try:
                duration = float(input("Enter duration in seconds: "))
                recorder.start_recording(duration=duration)
            except ValueError:
                print("Invalid duration")
            
        elif choice == '5':
            print("\nStarting continuous recording. Press Ctrl+C to stop.")
            recorder.start_recording(duration=float('inf'))
            
        elif choice == '6':
            if not recorder.find_target_streams():
                print("\nNo target streams found.")
            
        elif choice == '7':
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