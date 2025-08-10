#!/usr/bin/env python3
"""
Simple CSV to Database Populator
Populates video embeddings database with statistical features from EEG CSV files
This bypasses complex NeuroLM initialization and provides a working baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import sqlite3
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCSVToDatabase:
    """Simple CSV to database populator using statistical features as embeddings"""
    
    def __init__(self, data_dir: str, db_path: str = "video_embeddings.db"):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        
        # EEG processing parameters
        self.sampling_rate = 250
        self.n_channels = 6
        self.embedding_dim = 768  # Match NeuroLM-B dimension
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for video metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT UNIQUE NOT NULL,
                    video_url TEXT,
                    video_title TEXT,
                    embedding_data BLOB,
                    eeg_metrics TEXT,  -- JSON string of attention, engagement, etc.
                    experiment_date TEXT,
                    duration_seconds REAL,
                    embedding_stats TEXT,  -- JSON string of embedding statistics
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_id ON video_embeddings(video_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiment_date ON video_embeddings(experiment_date)')
            
            conn.commit()
            logger.info("✅ Database initialized successfully")
    
    def load_csv_file(self, csv_file: Path) -> Optional[np.ndarray]:
        """Load EEG data from CSV file"""
        try:
            logger.info(f"📄 Loading: {csv_file.name}")
            
            # Try different CSV reading approaches
            try:
                df = pd.read_csv(csv_file, skiprows=7, low_memory=False)
            except:
                try:
                    df = pd.read_csv(csv_file, comment='#')
                except:
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
            if eeg_data.shape[0] < 1000:  # At least 4 seconds of data
                logger.warning(f"⚠️ Insufficient data in {csv_file.name}: {eeg_data.shape[0]} samples")
                return None
            
            # Ensure we have the expected number of channels
            if eeg_data.shape[1] != self.n_channels:
                logger.warning(f"⚠️ Expected {self.n_channels} channels, got {eeg_data.shape[1]} in {csv_file.name}")
                if eeg_data.shape[1] > self.n_channels:
                    eeg_data = eeg_data[:, :self.n_channels]
                    logger.info(f"📈 Truncated to {self.n_channels} channels")
                else:
                    return None
            
            return eeg_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load {csv_file}: {e}")
            return None
    
    def extract_statistical_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """Extract statistical features from EEG data to create embeddings"""
        features = []
        
        # Time-domain features for each channel
        for ch in range(eeg_data.shape[1]):
            channel_data = eeg_data[:, ch]
            
            # Basic statistics
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.ptp(channel_data),  # peak-to-peak
                np.mean(np.abs(channel_data))  # mean absolute value
            ])
            
            # Higher-order statistics
            features.extend([
                np.sqrt(np.mean(channel_data**2)),  # RMS
                np.sum(np.abs(np.diff(channel_data))),  # Total variation
                len(np.where(np.diff(np.sign(channel_data)))[0]),  # Zero crossings
            ])
        
        # Cross-channel features
        # Correlation matrix (upper triangle)
        corr_matrix = np.corrcoef(eeg_data.T)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        features.extend(upper_triangle.tolist())
        
        # Global features across all channels
        features.extend([
            np.mean(eeg_data),
            np.std(eeg_data),
            np.var(eeg_data),
            np.mean(np.std(eeg_data, axis=0)),  # Mean channel std
            np.std(np.mean(eeg_data, axis=0)),  # Std of channel means
        ])
        
        # Convert to numpy array and pad/truncate to match embedding dimension
        features = np.array(features, dtype=np.float32)
        
        if len(features) < self.embedding_dim:
            # Pad with zeros if too short
            padded_features = np.zeros(self.embedding_dim, dtype=np.float32)
            padded_features[:len(features)] = features
            features = padded_features
        elif len(features) > self.embedding_dim:
            # Truncate if too long
            features = features[:self.embedding_dim]
        
        logger.info(f"✅ Extracted {len(features)} statistical features")
        return features.reshape(1, -1)  # Shape: (1, embedding_dim)
    
    def calculate_eeg_metrics(self, eeg_data: np.ndarray) -> Dict:
        """Calculate EEG-based attention and engagement metrics"""
        metrics = {}
        
        # Basic signal quality metrics
        metrics['signal_quality'] = {
            'mean_amplitude': float(np.mean(np.abs(eeg_data))),
            'signal_to_noise_ratio': float(np.mean(eeg_data) / np.std(eeg_data)) if np.std(eeg_data) > 0 else 0,
            'channel_correlation': float(np.mean(np.corrcoef(eeg_data.T)[np.triu_indices_from(np.corrcoef(eeg_data.T), k=1)]))
        }
        
        # Simulated attention metrics (in real system, these would come from frequency analysis)
        metrics['attention'] = {
            'focus_level': float(np.clip(np.random.normal(0.6, 0.2), 0, 1)),  # Placeholder
            'alertness': float(np.clip(np.random.normal(0.7, 0.15), 0, 1)),  # Placeholder
            'cognitive_load': float(np.clip(np.random.normal(0.5, 0.2), 0, 1))  # Placeholder
        }
        
        # Engagement metrics
        metrics['engagement'] = {
            'emotional_engagement': float(np.clip(np.random.normal(0.6, 0.2), 0, 1)),  # Placeholder
            'mental_effort': float(np.clip(np.random.normal(0.5, 0.2), 0, 1)),  # Placeholder
            'fatigue_level': float(np.clip(np.random.normal(0.3, 0.15), 0, 1))  # Placeholder
        }
        
        return metrics
    
    def store_video_embedding(self, video_id: str, embedding: np.ndarray, 
                            video_url: str = None, video_title: str = None,
                            eeg_metrics: Dict = None, duration_seconds: float = None) -> bool:
        """Store video embedding and metadata in the database"""
        try:
            # Calculate embedding statistics
            embedding_stats = {
                'mean': float(np.mean(embedding)),
                'std': float(np.std(embedding)),
                'min': float(np.min(embedding)),
                'max': float(np.max(embedding)),
                'norm': float(np.linalg.norm(embedding))
            }
            
            experiment_date = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if video already exists
                cursor.execute('SELECT id FROM video_embeddings WHERE video_id = ?', (video_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute('''
                        UPDATE video_embeddings 
                        SET embedding_data = ?, eeg_metrics = ?, embedding_stats = ?,
                            video_url = ?, video_title = ?, experiment_date = ?, duration_seconds = ?
                        WHERE video_id = ?
                    ''', (
                        pickle.dumps(embedding),
                        json.dumps(eeg_metrics) if eeg_metrics else None,
                        json.dumps(embedding_stats),
                        video_url, video_title, experiment_date, duration_seconds,
                        video_id
                    ))
                    logger.info(f"✅ Updated embedding for video: {video_id}")
                else:
                    # Insert new record
                    cursor.execute('''
                        INSERT INTO video_embeddings 
                        (video_id, video_url, video_title, embedding_data, eeg_metrics, 
                         experiment_date, duration_seconds, embedding_stats)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        video_id, video_url, video_title,
                        pickle.dumps(embedding),
                        json.dumps(eeg_metrics) if eeg_metrics else None,
                        experiment_date, duration_seconds,
                        json.dumps(embedding_stats)
                    ))
                    logger.info(f"✅ Stored new embedding for video: {video_id}")
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to store video embedding: {e}")
            return False
    
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
                
                # Extract statistical features as embeddings
                embedding = self.extract_statistical_features(eeg_data)
                
                # Calculate EEG metrics
                eeg_metrics = self.calculate_eeg_metrics(eeg_data)
                
                # Prepare metadata
                session_id = csv_file.stem
                duration_seconds = eeg_data.shape[0] / self.sampling_rate
                video_title = f"EEG Session {session_id}"
                video_url = f"file://{csv_file.absolute()}"
                
                # Store in database
                success = self.store_video_embedding(
                    video_id=session_id,
                    embedding=embedding,
                    video_url=video_url,
                    video_title=video_title,
                    eeg_metrics=eeg_metrics,
                    duration_seconds=duration_seconds
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
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM video_embeddings')
                total_videos = cursor.fetchone()[0]
                
                cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM video_embeddings')
                date_range = cursor.fetchone()
                
                cursor.execute('SELECT AVG(duration_seconds) FROM video_embeddings WHERE duration_seconds IS NOT NULL')
                avg_duration = cursor.fetchone()[0]
                
                return {
                    'total_videos': total_videos,
                    'date_range': f"{date_range[0]} to {date_range[1]}" if date_range[0] else "N/A",
                    'average_duration_seconds': round(avg_duration, 2) if avg_duration else 0,
                    'database_path': str(self.db_path)
                }
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {}

def main():
    """Main function to populate the database"""
    parser = argparse.ArgumentParser(description="Simple CSV to database populator")
    parser.add_argument("--data_dir", default="/Users/e.baena/Desktop/data", 
                       help="Directory containing CSV files")
    parser.add_argument("--db_path", default="video_embeddings.db", 
                       help="Path to the video embeddings database")
    
    args = parser.parse_args()
    
    logger.info("🧠 Simple CSV to Video Embeddings Database Populator")
    logger.info("=" * 60)
    
    # Initialize populator
    populator = SimpleCSVToDatabase(args.data_dir, args.db_path)
    
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
