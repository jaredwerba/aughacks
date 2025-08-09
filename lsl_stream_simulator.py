#!/usr/bin/env python3
"""
LSL Stream Simulator for NeuroLM Real-Time Testing
==================================================
Simulates an OpenBCI LSL stream by retransmitting CSV EEG data
Perfect for testing the real-time NeuroLM system without hardware
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Optional, List
import argparse
from datetime import datetime

try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    print("⚠️ pylsl not available. Install with: pip install pylsl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSLStreamSimulator:
    """Simulates OpenBCI LSL stream from CSV data"""
    
    def __init__(self, csv_file: str, sampling_rate: int = 250):
        self.csv_file = csv_file
        self.sampling_rate = sampling_rate
        self.channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
        self.outlet: Optional[StreamOutlet] = None
        self.eeg_data: Optional[np.ndarray] = None
        
        if not LSL_AVAILABLE:
            raise ImportError("pylsl is required for LSL streaming. Install with: pip install pylsl")
    
    def load_csv_data(self) -> bool:
        """Load EEG data from CSV file"""
        try:
            logger.info(f"📂 Loading EEG data from: {self.csv_file}")
            
            # Read file line by line to handle inconsistent formats
            valid_lines = []
            with open(self.csv_file, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    # Skip header line
                    if 'unix_timestamp' in line.lower() or 'timestamp' in line.lower():
                        logger.info("Skipping header line")
                        continue
                    
                    # Split and check if we have enough columns
                    parts = line.split(',')
                    if len(parts) >= 11:  # timestamps + 8 EEG channels
                        # Extract last 8 values (EEG channels)
                        eeg_values = parts[-8:]
                        try:
                            # Convert to float
                            eeg_floats = [float(val) for val in eeg_values]
                            valid_lines.append(eeg_floats)
                        except ValueError:
                            # Skip lines with non-numeric EEG data
                            continue
            
            if not valid_lines:
                logger.error("No valid EEG data lines found")
                return False
            
            # Convert to numpy array
            self.eeg_data = np.array(valid_lines)
            
            logger.info(f"✅ Loaded {self.eeg_data.shape[0]} samples with {self.eeg_data.shape[1]} channels")
            logger.info(f"📊 Data shape: {self.eeg_data.shape}")
            logger.info(f"📈 Data range: [{self.eeg_data.min():.2f}, {self.eeg_data.max():.2f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV data: {e}")
            return False
    
    def create_lsl_stream(self) -> bool:
        """Create LSL stream outlet"""
        try:
            logger.info("🌐 Creating LSL stream outlet...")
            
            # Create stream info
            info = StreamInfo(
                name='OpenBCI_EEG_Simulator',
                type='EEG',
                channel_count=8,
                nominal_srate=self.sampling_rate,
                channel_format='float32',
                source_id='neurolm_simulator'
            )
            
            # Add channel information
            channels = info.desc().append_child("channels")
            for i, ch_name in enumerate(self.channel_names):
                ch = channels.append_child("channel")
                ch.append_child_value("label", ch_name)
                ch.append_child_value("unit", "microvolts")
                ch.append_child_value("type", "EEG")
            
            # Create outlet
            self.outlet = StreamOutlet(info)
            
            logger.info("✅ LSL stream outlet created successfully")
            logger.info(f"📡 Stream name: OpenBCI_EEG_Simulator")
            logger.info(f"🔢 Channels: {len(self.channel_names)}")
            logger.info(f"⚡ Sampling rate: {self.sampling_rate} Hz")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating LSL stream: {e}")
            return False
    
    def start_streaming(self, loop: bool = False, speed_multiplier: float = 1.0) -> None:
        """Start streaming EEG data"""
        if self.eeg_data is None or self.outlet is None:
            logger.error("❌ Data not loaded or stream not created")
            return
        
        logger.info("🚀 Starting EEG data streaming...")
        logger.info(f"⏱️ Speed multiplier: {speed_multiplier}x")
        logger.info(f"🔄 Loop mode: {'ON' if loop else 'OFF'}")
        logger.info("📡 Stream is now active - connect your NeuroLM system!")
        logger.info("=" * 60)
        
        sample_interval = (1.0 / self.sampling_rate) / speed_multiplier
        sample_count = 0
        
        try:
            while True:
                for i in range(len(self.eeg_data)):
                    # Get current sample
                    sample = self.eeg_data[i].astype(np.float32)
                    
                    # Send sample via LSL
                    self.outlet.push_sample(sample)
                    
                    sample_count += 1
                    
                    # Log progress every 5 seconds
                    if sample_count % (self.sampling_rate * 5) == 0:
                        elapsed_time = sample_count / self.sampling_rate
                        logger.info(f"📊 Streamed {sample_count} samples ({elapsed_time:.1f}s of data)")
                    
                    # Wait for next sample
                    time.sleep(sample_interval)
                
                if not loop:
                    break
                else:
                    logger.info("🔄 Restarting data stream (loop mode)")
                    sample_count = 0
        
        except KeyboardInterrupt:
            logger.info("\n⏹️ Streaming stopped by user")
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
        finally:
            logger.info("🛑 Stream simulation ended")
            logger.info(f"📊 Total samples streamed: {sample_count}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='LSL Stream Simulator for NeuroLM Testing')
    parser.add_argument('csv_file', help='Path to CSV file with EEG data')
    parser.add_argument('--sampling-rate', '-sr', type=int, default=250, 
                       help='Sampling rate in Hz (default: 250)')
    parser.add_argument('--loop', '-l', action='store_true',
                       help='Loop the data continuously')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                       help='Speed multiplier (default: 1.0 = real-time)')
    
    args = parser.parse_args()
    
    print("🎬 LSL Stream Simulator for NeuroLM")
    print("=" * 50)
    print("🧠 Simulating OpenBCI EEG stream from CSV data")
    print("⚡ Perfect for testing real-time NeuroLM system")
    print("=" * 50)
    
    # Create simulator
    simulator = LSLStreamSimulator(args.csv_file, args.sampling_rate)
    
    # Load data
    if not simulator.load_csv_data():
        return 1
    
    # Create LSL stream
    if not simulator.create_lsl_stream():
        return 1
    
    print("\n🎯 Ready to start streaming!")
    print("💡 Start your NeuroLM real-time system in another terminal:")
    print("   python lsl_neurolm_realtime_system.py")
    print("\n⏱️ Starting in 3 seconds...")
    time.sleep(3)
    
    # Start streaming
    simulator.start_streaming(loop=args.loop, speed_multiplier=args.speed)
    
    return 0

if __name__ == "__main__":
    exit(main())
