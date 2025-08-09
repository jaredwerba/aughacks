#!/usr/bin/env python3

import numpy as np
from pylsl import StreamInlet, resolve_streams
import time
import csv
from datetime import datetime
import os

def connect_to_openbci_stream():
    print("Looking for OpenBCI LSL stream...")
    
    streams = resolve_streams()
    
    if not streams:
        print("No LSL streams found. Make sure OpenBCI is streaming via LSL.")
        return None
    
    print(f"\nFound {len(streams)} stream(s):")
    for i, stream in enumerate(streams):
        print(f"{i}: {stream.name()} - {stream.type()} - {stream.channel_count()} channels @ {stream.nominal_srate()} Hz")
    
    openbci_stream = None
    for stream in streams:
        if 'OpenBCI' in stream.name() or 'obci' in stream.name().lower() or stream.type() == 'EEG':
            openbci_stream = stream
            print(f"\nAutomatic selection: {stream.name()}")
            break
    
    if not openbci_stream:
        if len(streams) == 1:
            openbci_stream = streams[0]
            print(f"\nUsing the only available stream: {streams[0].name()}")
        else:
            print("\nNo OpenBCI stream detected. Please select a stream manually:")
            selection = int(input("Enter stream number: "))
            openbci_stream = streams[selection]
    
    inlet = StreamInlet(openbci_stream)
    
    info = inlet.info()
    print(f"\nConnected to: {info.name()}")
    print(f"Stream type: {info.type()}")
    print(f"Channel count: {info.channel_count()}")
    print(f"Sampling rate: {info.nominal_srate()} Hz")
    print(f"Stream ID: {info.source_id()}")
    
    return inlet

def read_data(inlet, duration=10, save_to_file=True):
    print(f"\nReading data for {duration} seconds...")
    print("Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    sample_count = 0
    
    # Create data directory if it doesn't exist
    data_dir = "lsl_data"
    if save_to_file and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Generate filename with timestamp
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(data_dir, f"lsl_stream_{session_timestamp}.csv") if save_to_file else None
    
    # Get stream info for CSV headers
    info = inlet.info()
    num_channels = info.channel_count()
    
    # Open CSV file and write headers
    csv_file = None
    csv_writer = None
    if save_to_file:
        csv_file = open(filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        # Write metadata header
        csv_writer.writerow(['# OpenBCI LSL Stream Data'])
        csv_writer.writerow([f'# Session Start: {datetime.now().isoformat()}'])
        csv_writer.writerow([f'# Stream Name: {info.name()}'])
        csv_writer.writerow([f'# Stream Type: {info.type()}'])
        csv_writer.writerow([f'# Sampling Rate: {info.nominal_srate()} Hz'])
        csv_writer.writerow([f'# Channel Count: {num_channels}'])
        csv_writer.writerow(['# Data Format: unix_timestamp, lsl_timestamp, system_time_iso, channel_1, channel_2, ...'])
        
        # Write column headers
        headers = ['unix_timestamp', 'lsl_timestamp', 'system_time_iso'] + [f'channel_{i+1}' for i in range(num_channels)]
        csv_writer.writerow(headers)
        
        print(f"Saving data to: {filename}")
    
    try:
        while (time.time() - start_time) < duration:
            sample, lsl_timestamp = inlet.pull_sample(timeout=1.0)
            
            if sample:
                sample_count += 1
                unix_timestamp = time.time()
                system_time_iso = datetime.now().isoformat()
                
                # Save to file
                if save_to_file and csv_writer:
                    row = [unix_timestamp, lsl_timestamp, system_time_iso] + sample
                    csv_writer.writerow(row)
                
                # Print periodic updates
                if sample_count % 100 == 1:
                    print(f"Sample {sample_count} at {lsl_timestamp:.3f}:")
                    print(f"  Channels 1-4: {np.array(sample[:4])}")
                    if len(sample) > 4:
                        print(f"  Channels 5-8: {np.array(sample[4:8])}")
                    print()
                    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if csv_file:
            csv_file.close()
    
    print(f"\nTotal samples received: {sample_count}")
    print(f"Effective sampling rate: {sample_count / (time.time() - start_time):.2f} Hz")
    
    if save_to_file:
        print(f"Data saved to: {filename}")
        return filename
    
    return None

def main():
    print("OpenBCI LSL Stream Reader")
    print("=" * 40)
    
    inlet = connect_to_openbci_stream()
    
    if inlet:
        print("\nSuccessfully connected to OpenBCI stream!")
        
        while True:
            print("\nOptions:")
            print("1. Read data for 10 seconds (with saving)")
            print("2. Read data continuously (with saving)")
            print("3. Read data for custom duration (with saving)")
            print("4. Read without saving (10 seconds)")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                read_data(inlet, duration=10, save_to_file=True)
            elif choice == '2':
                read_data(inlet, duration=float('inf'), save_to_file=True)
            elif choice == '3':
                duration = float(input("Enter duration in seconds: "))
                read_data(inlet, duration=duration, save_to_file=True)
            elif choice == '4':
                read_data(inlet, duration=10, save_to_file=False)
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice")
    else:
        print("\nFailed to connect to any LSL stream")
        print("\nMake sure:")
        print("1. OpenBCI GUI is running")
        print("2. LSL streaming is enabled in OpenBCI GUI")
        print("3. Your firewall isn't blocking LSL connections")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("Error: pylsl library not installed")
        print("Install it with: pip install pylsl")
    except Exception as e:
        print(f"Error: {e}")