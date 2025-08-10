#!/usr/bin/env python3
"""
Fragment-based CSV to Database Populator
Populates video embeddings database with statistical features from EEG CSV files
Creates multiple embeddings per video based on time fragments of specified duration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import sqlite3
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FragmentCSVToDatabase:
    """Fragment-based CSV to database populator using statistical features as embeddings"""
    
    def __init__(self, data_dir: str, db_path: str = "video_embeddings_fragments.db", 
                 fragment_duration: float = 30.0, overlap_ratio: float = 0.5):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.fragment_duration = fragment_duration  # Duration in seconds
        self.overlap_ratio = overlap_ratio  # Overlap between fragments (0.0 = no overlap, 0.5 = 50% overlap)
        
        # EEG processing parameters
        self.sampling_rate = 250
        self.n_channels = 6
        self.embedding_dim = 768  # Match NeuroLM-B dimension
        
        # Calculate fragment parameters
        self.fragment_samples = int(self.fragment_duration * self.sampling_rate)
        self.step_samples = int(self.fragment_samples * (1 - self.overlap_ratio))
        
        logger.info(f"📊 Fragment configuration:")
        logger.info(f"   🕐 Duration: {self.fragment_duration} seconds")
        logger.info(f"   📏 Samples per fragment: {self.fragment_samples}")
        logger.info(f"   👥 Overlap: {self.overlap_ratio*100:.1f}%")
        logger.info(f"   📐 Step size: {self.step_samples} samples")
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for video fragments metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    fragment_id TEXT UNIQUE NOT NULL,
                    fragment_index INTEGER NOT NULL,
                    start_time_seconds REAL NOT NULL,
                    end_time_seconds REAL NOT NULL,
                    video_url TEXT,
                    video_title TEXT,
                    embedding_data BLOB,
                    eeg_metrics TEXT,  -- JSON string of attention, engagement, etc.
                    experiment_date TEXT,
                    fragment_duration_seconds REAL,
                    total_video_duration_seconds REAL,
                    embedding_stats TEXT,  -- JSON string of embedding statistics
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_id ON video_embeddings(video_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fragment_id ON video_embeddings(fragment_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_start_time ON video_embeddings(start_time_seconds)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiment_date ON video_embeddings(experiment_date)')
            
            conn.commit()
            logger.info("✅ Fragment database initialized successfully")
    
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
            
            # Check if we have enough data for at least one fragment
            if eeg_data.shape[0] < self.fragment_samples:
                logger.warning(f"⚠️ Insufficient data in {csv_file.name}: {eeg_data.shape[0]} < {self.fragment_samples} samples")
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
    
    def extract_fragments(self, eeg_data: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """Extract time fragments from EEG data"""
        fragments = []
        total_samples = eeg_data.shape[0]
        
        # Calculate fragment positions
        start_positions = range(0, total_samples - self.fragment_samples + 1, self.step_samples)
        
        for i, start_pos in enumerate(start_positions):
            end_pos = start_pos + self.fragment_samples
            
            # Extract fragment
            fragment_data = eeg_data[start_pos:end_pos, :]
            
            # Calculate time boundaries
            start_time = start_pos / self.sampling_rate
            end_time = end_pos / self.sampling_rate
            
            fragments.append((fragment_data, start_time, end_time))
        
        logger.info(f"🔪 Extracted {len(fragments)} fragments from {total_samples} samples")
        return fragments
    
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
        
        # Temporal dynamics (for fragments)
        # First and second derivatives
        if eeg_data.shape[0] > 1:
            first_diff = np.diff(eeg_data, axis=0)
            features.extend([
                np.mean(first_diff),
                np.std(first_diff),
                np.mean(np.abs(first_diff))
            ])
            
            if eeg_data.shape[0] > 2:
                second_diff = np.diff(first_diff, axis=0)
                features.extend([
                    np.mean(second_diff),
                    np.std(second_diff),
                    np.mean(np.abs(second_diff))
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
        
        return features.reshape(1, -1)  # Shape: (1, embedding_dim)
    
    def calculate_eeg_metrics(self, eeg_data: np.ndarray, start_time: float, end_time: float) -> Dict:
        """Calculate EEG-based attention and engagement metrics for a fragment"""
        metrics = {}
        
        # Fragment-specific info
        metrics['fragment_info'] = {
            'start_time': float(start_time),
            'end_time': float(end_time),
            'duration': float(end_time - start_time),
            'n_samples': int(eeg_data.shape[0])
        }
        
        # Basic signal quality metrics
        metrics['signal_quality'] = {
            'mean_amplitude': float(np.mean(np.abs(eeg_data))),
            'signal_to_noise_ratio': float(np.mean(eeg_data) / np.std(eeg_data)) if np.std(eeg_data) > 0 else 0,
            'channel_correlation': float(np.mean(np.corrcoef(eeg_data.T)[np.triu_indices_from(np.corrcoef(eeg_data.T), k=1)]))
        }
        
        # Temporal stability within fragment
        if eeg_data.shape[0] > 1:
            temporal_stability = 1.0 / (1.0 + np.mean(np.std(np.diff(eeg_data, axis=0), axis=0)))
            metrics['signal_quality']['temporal_stability'] = float(temporal_stability)
        
        # Simulated attention metrics (in real system, these would come from frequency analysis)
        # Add some temporal variation based on fragment position
        time_factor = np.sin(start_time * 0.1) * 0.1  # Slow oscillation
        
        metrics['attention'] = {
            'focus_level': float(np.clip(np.random.normal(0.6 + time_factor, 0.2), 0, 1)),
            'alertness': float(np.clip(np.random.normal(0.7 + time_factor * 0.5, 0.15), 0, 1)),
            'cognitive_load': float(np.clip(np.random.normal(0.5 - time_factor * 0.3, 0.2), 0, 1))
        }
        
        # Engagement metrics
        metrics['engagement'] = {
            'emotional_engagement': float(np.clip(np.random.normal(0.6 + time_factor * 0.2, 0.2), 0, 1)),
            'mental_effort': float(np.clip(np.random.normal(0.5 + time_factor * 0.4, 0.2), 0, 1)),
            'fatigue_level': float(np.clip(np.random.normal(0.3 - time_factor * 0.1, 0.15), 0, 1))
        }
        
        return metrics
    
    def store_fragment_embedding(self, video_id: str, fragment_index: int, 
                                embedding: np.ndarray, start_time: float, end_time: float,
                                video_url: str = None, video_title: str = None,
                                eeg_metrics: Dict = None, total_duration: float = None) -> bool:
        """Store fragment embedding and metadata in the database"""
        try:
            # Create unique fragment ID
            fragment_id = f"{video_id}_fragment_{fragment_index:04d}"
            
            # Calculate embedding statistics
            embedding_stats = {
                'mean': float(np.mean(embedding)),
                'std': float(np.std(embedding)),
                'min': float(np.min(embedding)),
                'max': float(np.max(embedding)),
                'norm': float(np.linalg.norm(embedding))
            }
            
            experiment_date = datetime.now().isoformat()
            fragment_duration = end_time - start_time
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if fragment already exists
                cursor.execute('SELECT id FROM video_embeddings WHERE fragment_id = ?', (fragment_id,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute('''
                        UPDATE video_embeddings 
                        SET embedding_data = ?, eeg_metrics = ?, embedding_stats = ?,
                            video_url = ?, video_title = ?, experiment_date = ?, 
                            fragment_duration_seconds = ?, total_video_duration_seconds = ?
                        WHERE fragment_id = ?
                    ''', (
                        pickle.dumps(embedding),
                        json.dumps(eeg_metrics) if eeg_metrics else None,
                        json.dumps(embedding_stats),
                        video_url, video_title, experiment_date,
                        fragment_duration, total_duration,
                        fragment_id
                    ))
                else:
                    # Insert new record
                    cursor.execute('''
                        INSERT INTO video_embeddings 
                        (video_id, fragment_id, fragment_index, start_time_seconds, end_time_seconds,
                         video_url, video_title, embedding_data, eeg_metrics, experiment_date,
                         fragment_duration_seconds, total_video_duration_seconds, embedding_stats)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        video_id, fragment_id, fragment_index, start_time, end_time,
                        video_url, video_title,
                        pickle.dumps(embedding),
                        json.dumps(eeg_metrics) if eeg_metrics else None,
                        experiment_date, fragment_duration, total_duration,
                        json.dumps(embedding_stats)
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to store fragment embedding: {e}")
            return False
    
    def process_csv_files(self) -> Dict[str, Dict]:
        """Process all CSV files in the data directory"""
        logger.info(f"📂 Processing CSV files in: {self.data_dir}")
        
        # Find all LSL stream CSV files
        csv_files = list(self.data_dir.glob("lsl_stream_*.csv"))
        
        if not csv_files:
            logger.error(f"❌ No LSL stream CSV files found in {self.data_dir}")
            return {}
        
        logger.info(f"📁 Found {len(csv_files)} CSV files to process")
        
        results = {}
        total_fragments_processed = 0
        
        for csv_file in sorted(csv_files):
            try:
                logger.info(f"\n🎬 Processing file: {csv_file.name}")
                
                # Load EEG data
                eeg_data = self.load_csv_file(csv_file)
                if eeg_data is None:
                    results[csv_file.name] = {'success': False, 'fragments': 0, 'error': 'Failed to load data'}
                    continue
                
                # Extract fragments
                fragments = self.extract_fragments(eeg_data)
                
                if not fragments:
                    results[csv_file.name] = {'success': False, 'fragments': 0, 'error': 'No fragments extracted'}
                    continue
                
                # Prepare metadata
                session_id = csv_file.stem
                total_duration = eeg_data.shape[0] / self.sampling_rate
                video_title = f"EEG Session {session_id}"
                video_url = f"file://{csv_file.absolute()}"
                
                # Process each fragment
                successful_fragments = 0
                for fragment_index, (fragment_data, start_time, end_time) in enumerate(fragments):
                    try:
                        # Extract statistical features as embeddings
                        embedding = self.extract_statistical_features(fragment_data)
                        
                        # Calculate EEG metrics for this fragment
                        eeg_metrics = self.calculate_eeg_metrics(fragment_data, start_time, end_time)
                        
                        # Store in database
                        success = self.store_fragment_embedding(
                            video_id=session_id,
                            fragment_index=fragment_index,
                            embedding=embedding,
                            start_time=start_time,
                            end_time=end_time,
                            video_url=video_url,
                            video_title=video_title,
                            eeg_metrics=eeg_metrics,
                            total_duration=total_duration
                        )
                        
                        if success:
                            successful_fragments += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing fragment {fragment_index}: {e}")
                
                results[csv_file.name] = {
                    'success': successful_fragments > 0,
                    'fragments': successful_fragments,
                    'total_fragments': len(fragments),
                    'total_duration': total_duration
                }
                
                total_fragments_processed += successful_fragments
                logger.info(f"✅ Processed {successful_fragments}/{len(fragments)} fragments for {csv_file.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {csv_file.name}: {e}")
                results[csv_file.name] = {'success': False, 'fragments': 0, 'error': str(e)}
        
        logger.info(f"\n🎯 Processing complete: {total_fragments_processed} total fragments processed")
        return results
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the populated database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total fragments
                cursor.execute('SELECT COUNT(*) FROM video_embeddings')
                total_fragments = cursor.fetchone()[0]
                
                # Unique videos
                cursor.execute('SELECT COUNT(DISTINCT video_id) FROM video_embeddings')
                unique_videos = cursor.fetchone()[0]
                
                # Average fragments per video
                cursor.execute('SELECT AVG(fragment_count) FROM (SELECT COUNT(*) as fragment_count FROM video_embeddings GROUP BY video_id)')
                avg_fragments_per_video = cursor.fetchone()[0]
                
                # Date range
                cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM video_embeddings')
                date_range = cursor.fetchone()
                
                # Average fragment duration
                cursor.execute('SELECT AVG(fragment_duration_seconds) FROM video_embeddings WHERE fragment_duration_seconds IS NOT NULL')
                avg_fragment_duration = cursor.fetchone()[0]
                
                return {
                    'total_fragments': total_fragments,
                    'unique_videos': unique_videos,
                    'avg_fragments_per_video': round(avg_fragments_per_video, 1) if avg_fragments_per_video else 0,
                    'date_range': f"{date_range[0]} to {date_range[1]}" if date_range[0] else "N/A",
                    'avg_fragment_duration_seconds': round(avg_fragment_duration, 2) if avg_fragment_duration else 0,
                    'database_path': str(self.db_path)
                }
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {}

def main():
    """Main function to populate the database with fragments"""
    parser = argparse.ArgumentParser(description="Fragment-based CSV to database populator")
    parser.add_argument("--data_dir", default="/Users/e.baena/Desktop/data", 
                       help="Directory containing CSV files")
    parser.add_argument("--db_path", default="video_embeddings_fragments.db", 
                       help="Path to the video embeddings database")
    parser.add_argument("--fragment_duration", type=float, default=30.0,
                       help="Duration of each fragment in seconds (default: 30.0)")
    parser.add_argument("--overlap_ratio", type=float, default=0.5,
                       help="Overlap ratio between fragments (0.0-1.0, default: 0.5)")
    
    args = parser.parse_args()
    
    logger.info("🧠 Fragment-based CSV to Video Embeddings Database Populator")
    logger.info("=" * 70)
    
    # Initialize populator
    populator = FragmentCSVToDatabase(
        args.data_dir, 
        args.db_path, 
        args.fragment_duration,
        args.overlap_ratio
    )
    
    # Process CSV files
    results = populator.process_csv_files()
    
    # Show results
    logger.info("\n📊 Processing Results:")
    logger.info("-" * 40)
    
    total_successful_files = sum(1 for r in results.values() if r.get('success', False))
    total_files = len(results)
    total_fragments = sum(r.get('fragments', 0) for r in results.values())
    
    for filename, result in results.items():
        if result.get('success', False):
            status = f"✅ SUCCESS: {result['fragments']}/{result.get('total_fragments', '?')} fragments"
        else:
            status = f"❌ FAILED: {result.get('error', 'Unknown error')}"
        logger.info(f"{status} - {filename}")
    
    logger.info(f"\n🎯 Final Summary:")
    logger.info(f"   📁 Files: {total_successful_files}/{total_files} processed successfully")
    logger.info(f"   🔪 Fragments: {total_fragments} total fragments created")
    
    # Show database statistics
    if total_fragments > 0:
        logger.info("\n📈 Database Statistics:")
        stats = populator.get_database_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("\n✅ Fragment-based database population complete!")

if __name__ == "__main__":
    main()
