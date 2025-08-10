#!/usr/bin/env python3
"""
Populate Video Embeddings Database from CSV Files
Processes EEG CSV files and stores NeuroLM embeddings in the vector database
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Import our custom modules
from neurolm_embedding_extractor import NeuroLMEmbeddingExtractor
from vector_database import VideoEmbeddingVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVToEmbeddingsPopulator:
    """Populate video embeddings database from CSV EEG data"""
    
    def __init__(self, data_dir: str, db_path: str = "video_embeddings.db"):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.embedding_extractor = NeuroLMEmbeddingExtractor(db_path)
        self.vector_db = VideoEmbeddingVectorDB(db_path)
        
        # EEG processing parameters (matching the existing codebase)
        self.sampling_rate = 250
        self.window_size = 1000  # 4 seconds
        self.n_channels = 6
        
    async def initialize(self) -> bool:
        """Initialize NeuroLM components"""
        logger.info("🔧 Initializing NeuroLM embedding extractor...")
        success = await self.embedding_extractor.initialize_neurolm()
        if success:
            logger.info("✅ NeuroLM initialized successfully!")
        else:
            logger.warning("⚠️ NeuroLM initialization failed, using random weights")
        return success
    
    def load_csv_file(self, csv_file: Path) -> Optional[np.ndarray]:
        """Load EEG data from CSV file"""
        try:
            logger.info(f"📄 Loading: {csv_file.name}")
            
            # Try different CSV reading approaches based on existing codebase
            try:
                # First try: skip comment lines (from eeg_classification_with_neurolm_b.py)
                df = pd.read_csv(csv_file, skiprows=7, low_memory=False)
            except:
                try:
                    # Second try: use comment parameter (from neurolm_b_checkpoint_system.py)
                    df = pd.read_csv(csv_file, comment='#')
                except:
                    # Third try: read normally
                    df = pd.read_csv(csv_file, low_memory=False)
            
            logger.info(f"📊 CSV columns: {list(df.columns)}")
            
            # Extract EEG channels
            eeg_columns = [col for col in df.columns if col.startswith('channel_')]
            logger.info(f"🎯 EEG channels found: {eeg_columns}")
            
            if not eeg_columns:
                logger.warning(f"⚠️ No EEG channels found in {csv_file.name}")
                return None
            
            # Get EEG data
            eeg_data = df[eeg_columns].values
            logger.info(f"📈 Raw EEG data shape: {eeg_data.shape}")
            
            # Clean data - remove NaN values
            eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)]
            logger.info(f"📈 Clean EEG data shape: {eeg_data.shape}")
            
            # Check if we have enough data
            if eeg_data.shape[0] < self.window_size:
                logger.warning(f"⚠️ Insufficient data in {csv_file.name}: {eeg_data.shape[0]} < {self.window_size}")
                return None
            
            # Ensure we have the expected number of channels
            if eeg_data.shape[1] != self.n_channels:
                logger.warning(f"⚠️ Expected {self.n_channels} channels, got {eeg_data.shape[1]} in {csv_file.name}")
                # If we have more channels, take the first 6
                if eeg_data.shape[1] > self.n_channels:
                    eeg_data = eeg_data[:, :self.n_channels]
                    logger.info(f"📈 Truncated to {self.n_channels} channels")
                else:
                    return None
            
            return eeg_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load {csv_file}: {e}")
            return None
    
    def extract_video_metadata(self, csv_file: Path, eeg_data: np.ndarray) -> Dict:
        """Extract metadata for the video/session"""
        session_id = csv_file.stem
        
        # Calculate duration (assuming 250 Hz sampling rate)
        duration_seconds = eeg_data.shape[0] / self.sampling_rate
        
        # Create video metadata
        metadata = {
            'video_id': session_id,
            'video_title': f"EEG Session {session_id}",
            'video_url': f"file://{csv_file.absolute()}",
            'duration_seconds': duration_seconds,
            'experiment_metrics': {
                'n_samples': int(eeg_data.shape[0]),
                'n_channels': int(eeg_data.shape[1]),
                'sampling_rate': self.sampling_rate,
                'file_size_mb': csv_file.stat().st_size / (1024 * 1024),
                'data_quality': {
                    'mean_amplitude': float(np.mean(eeg_data)),
                    'std_amplitude': float(np.std(eeg_data)),
                    'min_amplitude': float(np.min(eeg_data)),
                    'max_amplitude': float(np.max(eeg_data))
                }
            }
        }
        
        return metadata
    
    def process_csv_files(self) -> Dict[str, bool]:
        """Process all CSV files in the data directory"""
        logger.info(f"📂 Processing CSV files in: {self.data_dir}")
        
        # Find all LSL stream CSV files
        csv_files = list(self.data_dir.glob("lsl_stream_*.csv"))
        
        if not csv_files:
            logger.error(f"❌ No LSL stream CSV files found in {self.data_dir}")
            return {}
        
        logger.info(f"📁 Found {len(csv_files)} CSV files to process")
        
        results = {}
        processed_count = 0
        
        for csv_file in sorted(csv_files):
            try:
                logger.info(f"\n🎬 Processing file {processed_count + 1}/{len(csv_files)}: {csv_file.name}")
                
                # Load EEG data
                eeg_data = self.load_csv_file(csv_file)
                if eeg_data is None:
                    results[csv_file.name] = False
                    continue
                
                # Extract metadata
                metadata = self.extract_video_metadata(csv_file, eeg_data)
                
                # Process with embedding extractor
                success = self.embedding_extractor.process_video_experiment(
                    video_id=metadata['video_id'],
                    eeg_data=eeg_data,
                    video_url=metadata['video_url'],
                    video_title=metadata['video_title'],
                    experiment_metrics=metadata['experiment_metrics'],
                    duration_seconds=metadata['duration_seconds']
                )
                
                results[csv_file.name] = success
                if success:
                    processed_count += 1
                    logger.info(f"✅ Successfully processed {csv_file.name}")
                else:
                    logger.error(f"❌ Failed to process {csv_file.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {csv_file.name}: {e}")
                results[csv_file.name] = False
        
        logger.info(f"\n🎯 Processing complete: {processed_count}/{len(csv_files)} files successful")
        return results
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the populated database"""
        return self.vector_db.get_database_stats()

def main():
    """Main function to populate the database"""
    parser = argparse.ArgumentParser(description="Populate video embeddings database from CSV files")
    parser.add_argument("--data_dir", default="/Users/e.baena/Desktop/data", 
                       help="Directory containing CSV files")
    parser.add_argument("--db_path", default="video_embeddings.db", 
                       help="Path to the video embeddings database")
    
    args = parser.parse_args()
    
    logger.info("🧠 NeuroLM Video Embeddings Database Populator")
    logger.info("=" * 50)
    
    # Initialize populator
    populator = CSVToEmbeddingsPopulator(args.data_dir, args.db_path)
    
    # Initialize NeuroLM
    import asyncio
    asyncio.run(populator.initialize())
    
    # Process CSV files
    results = populator.process_csv_files()
    
    # Show results
    logger.info("\n📊 Processing Results:")
    logger.info("-" * 30)
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    for filename, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status}: {filename}")
    
    logger.info(f"\n🎯 Final Summary: {successful}/{total} files processed successfully")
    
    # Show database statistics
    if successful > 0:
        logger.info("\n📈 Database Statistics:")
        stats = populator.get_database_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("\n✅ Database population complete!")

if __name__ == "__main__":
    main()
