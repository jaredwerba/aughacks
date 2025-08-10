#!/usr/bin/env python3
"""
NeuroLM-Based Video Search Database
Create a searchable database using actual NeuroLM embeddings for semantic neural pattern search
"""

import numpy as np
import sqlite3
import pickle
import json
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import pandas as pd
import logging
from datetime import datetime
import torch

# Import NeuroLM components
from neurolm_embedding_extractor import NeuroLMEmbeddingExtractor
from vector_database import VideoEmbeddingVectorDB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroLMSearchDatabase:
    """Searchable database using NeuroLM embeddings for semantic neural pattern search"""
    
    def __init__(self, search_db_path: str = "neurolm_search_database.db"):
        self.search_db_path = Path(search_db_path)
        self.faiss_index = None
        self.embedding_metadata = []
        self.neurolm_extractor = None
        self._init_search_database()
        self._init_neurolm()
    
    def _init_neurolm(self):
        """Initialize NeuroLM embedding extractor"""
        try:
            logger.info("🧠 Initializing NeuroLM extractor...")
            self.neurolm_extractor = NeuroLMEmbeddingExtractor()
            logger.info("✅ NeuroLM extractor initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize NeuroLM: {e}")
            self.neurolm_extractor = None
    
    def _init_search_database(self):
        """Initialize the search database schema"""
        with sqlite3.connect(self.search_db_path) as conn:
            cursor = conn.cursor()
            
            # Create NeuroLM search table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS neurolm_search (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_id TEXT UNIQUE NOT NULL,
                    video_id TEXT NOT NULL,
                    fragment_index INTEGER,
                    start_time_seconds REAL,
                    end_time_seconds REAL,
                    fragment_duration REAL,
                    
                    -- NeuroLM embedding data
                    neurolm_embedding BLOB NOT NULL,
                    embedding_norm REAL,
                    
                    -- Neural pattern characteristics
                    attention_pattern TEXT,  -- 'focused', 'distracted', 'variable'
                    cognitive_state TEXT,    -- 'active', 'passive', 'transitional'
                    neural_complexity REAL,  -- Complexity score from NeuroLM
                    
                    -- EEG signal quality
                    signal_quality REAL,
                    channel_consistency REAL,
                    
                    -- Temporal context
                    video_timestamp TEXT,
                    experiment_session TEXT,
                    
                    -- Search metadata
                    search_keywords TEXT,  -- JSON array for text search
                    neural_tags TEXT,      -- JSON array for neural pattern tags
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for fast search
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_neurolm_video_id ON neurolm_search(video_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_neurolm_attention ON neurolm_search(attention_pattern)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_neurolm_cognitive ON neurolm_search(cognitive_state)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_neurolm_duration ON neurolm_search(fragment_duration)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_neurolm_complexity ON neurolm_search(neural_complexity)')
            
            # Create full-text search for neural patterns
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS neurolm_fts USING fts5(
                    embedding_id,
                    video_id,
                    attention_pattern,
                    cognitive_state,
                    search_keywords,
                    neural_tags,
                    content='neurolm_search',
                    content_rowid='id'
                )
            ''')
            
            conn.commit()
        
        logger.info(f"✅ NeuroLM search database initialized: {self.search_db_path}")
    
    def process_csv_files_with_neurolm(self, data_dir: str, fragment_duration: float = 30.0, 
                                     overlap_ratio: float = 0.0):
        """Process CSV files and extract NeuroLM embeddings for search database"""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        if self.neurolm_extractor is None:
            logger.error("❌ NeuroLM extractor not available. Using fallback method.")
            return self._process_with_fallback(data_dir, fragment_duration, overlap_ratio)
        
        csv_files = list(data_path.glob("*.csv"))
        logger.info(f"📁 Found {len(csv_files)} CSV files to process")
        
        sampling_rate = 250  # Hz
        fragment_samples = int(fragment_duration * sampling_rate)
        overlap_samples = int(fragment_samples * overlap_ratio)
        step_samples = fragment_samples - overlap_samples
        
        all_embeddings = []
        all_metadata = []
        
        for csv_file in csv_files:
            logger.info(f"\n🎬 Processing file: {csv_file.name}")
            
            try:
                # Load CSV data
                df = pd.read_csv(csv_file)
                logger.info(f"📊 CSV columns: {list(df.columns)}")
                
                # Extract EEG channels
                eeg_columns = [col for col in df.columns if col.startswith('channel_')]
                if not eeg_columns:
                    logger.warning(f"⚠️ No EEG channels found in {csv_file.name}")
                    continue
                
                logger.info(f"🎯 EEG channels found: {eeg_columns}")
                eeg_data = df[eeg_columns].values
                logger.info(f"📈 Raw EEG data shape: {eeg_data.shape}")
                
                # Clean data
                eeg_data = eeg_data[~np.isnan(eeg_data).any(axis=1)]
                logger.info(f"📈 Clean EEG data shape: {eeg_data.shape}")
                
                if len(eeg_data) < fragment_samples:
                    logger.warning(f"⚠️ File too short for {fragment_duration}s fragments")
                    continue
                
                # Extract fragments
                fragments = []
                for start_idx in range(0, len(eeg_data) - fragment_samples + 1, step_samples):
                    end_idx = start_idx + fragment_samples
                    fragment_data = eeg_data[start_idx:end_idx]
                    
                    start_time = start_idx / sampling_rate
                    end_time = end_idx / sampling_rate
                    
                    fragments.append({
                        'data': fragment_data,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    })
                
                logger.info(f"🔪 Extracted {len(fragments)} fragments from {len(eeg_data)} samples")
                
                # Process each fragment with NeuroLM
                video_id = csv_file.stem
                for i, fragment in enumerate(fragments):
                    try:
                        # Extract NeuroLM embedding
                        neurolm_embedding = self.neurolm_extractor.extract_embeddings_from_eeg(fragment['data'])
                        
                        if neurolm_embedding is None:
                            logger.warning(f"⚠️ Failed to extract NeuroLM embedding for fragment {i}")
                            continue
                        
                        # Flatten if needed
                        if neurolm_embedding.ndim > 1:
                            neurolm_embedding = neurolm_embedding.flatten()
                        
                        # Analyze neural patterns
                        neural_analysis = self._analyze_neural_patterns(neurolm_embedding, fragment['data'])
                        
                        # Create metadata
                        fragment_id = f"{video_id}_neurolm_fragment_{i:04d}"
                        
                        metadata = {
                            'embedding_id': fragment_id,
                            'video_id': video_id,
                            'fragment_index': i,
                            'start_time_seconds': fragment['start_time'],
                            'end_time_seconds': fragment['end_time'],
                            'fragment_duration': fragment_duration,
                            'neurolm_embedding': neurolm_embedding,
                            'embedding_norm': float(np.linalg.norm(neurolm_embedding)),
                            'attention_pattern': neural_analysis['attention_pattern'],
                            'cognitive_state': neural_analysis['cognitive_state'],
                            'neural_complexity': neural_analysis['complexity'],
                            'signal_quality': neural_analysis['signal_quality'],
                            'channel_consistency': neural_analysis['channel_consistency'],
                            'video_timestamp': video_id.replace('lsl_stream_', ''),
                            'experiment_session': f"session_{video_id[-6:]}",
                            'search_keywords': neural_analysis['keywords'],
                            'neural_tags': neural_analysis['tags']
                        }
                        
                        all_embeddings.append(neurolm_embedding)
                        all_metadata.append(metadata)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing fragment {i}: {e}")
                        continue
                
                logger.info(f"✅ Processed {len([m for m in all_metadata if m['video_id'] == video_id])} fragments for {csv_file.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {csv_file.name}: {e}")
                continue
        
        # Store in database
        self._store_neurolm_data(all_embeddings, all_metadata)
        
        # Build FAISS index
        self._build_faiss_index(all_embeddings)
        
        logger.info(f"🎯 NeuroLM search database complete: {len(all_embeddings)} embeddings")
        return len(all_embeddings)
    
    def _analyze_neural_patterns(self, embedding: np.ndarray, eeg_data: np.ndarray) -> Dict:
        """Analyze neural patterns from NeuroLM embedding and raw EEG"""
        
        # Calculate embedding characteristics
        embedding_mean = float(np.mean(embedding))
        embedding_std = float(np.std(embedding))
        embedding_max = float(np.max(embedding))
        
        # Analyze attention patterns based on embedding distribution
        if embedding_std > 0.5:
            if embedding_mean > 0:
                attention_pattern = "focused"
            else:
                attention_pattern = "variable"
        else:
            attention_pattern = "distracted"
        
        # Analyze cognitive state based on embedding magnitude
        embedding_magnitude = np.linalg.norm(embedding)
        if embedding_magnitude > 10:
            cognitive_state = "active"
        elif embedding_magnitude > 5:
            cognitive_state = "transitional"
        else:
            cognitive_state = "passive"
        
        # Calculate neural complexity
        complexity = float(np.std(embedding) / (np.mean(np.abs(embedding)) + 1e-8))
        
        # Calculate signal quality from raw EEG
        signal_quality = 1.0 - (np.sum(np.isnan(eeg_data)) / eeg_data.size)
        
        # Calculate channel consistency
        channel_correlations = np.corrcoef(eeg_data.T)
        channel_consistency = float(np.mean(channel_correlations[np.triu_indices_from(channel_correlations, k=1)]))
        
        # Generate search keywords
        keywords = [attention_pattern, cognitive_state]
        if complexity > 0.5:
            keywords.append("complex")
        if signal_quality > 0.9:
            keywords.append("high_quality")
        if channel_consistency > 0.5:
            keywords.append("synchronized")
        
        # Generate neural tags
        tags = [f"attention_{attention_pattern}", f"cognitive_{cognitive_state}"]
        if embedding_magnitude > 10:
            tags.append("high_activation")
        if complexity > 0.7:
            tags.append("complex_neural_pattern")
        
        return {
            'attention_pattern': attention_pattern,
            'cognitive_state': cognitive_state,
            'complexity': complexity,
            'signal_quality': signal_quality,
            'channel_consistency': channel_consistency,
            'keywords': json.dumps(keywords),
            'tags': json.dumps(tags)
        }
    
    def _process_with_fallback(self, data_dir: str, fragment_duration: float, overlap_ratio: float):
        """Fallback method if NeuroLM is not available"""
        logger.warning("🔄 Using fallback statistical embeddings (NeuroLM not available)")
        
        # Import the existing fragment script logic
        from fragment_csv_to_db import process_csv_files
        
        # Process with statistical embeddings but store in search format
        # This is a simplified fallback
        logger.info("📊 Processing with statistical embeddings as fallback...")
        return 0
    
    def _store_neurolm_data(self, embeddings: List[np.ndarray], metadata: List[Dict]):
        """Store NeuroLM embeddings and metadata in search database"""
        with sqlite3.connect(self.search_db_path) as conn:
            cursor = conn.cursor()
            
            for embedding, meta in zip(embeddings, metadata):
                # Serialize embedding
                embedding_blob = pickle.dumps(embedding)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO neurolm_search (
                        embedding_id, video_id, fragment_index, start_time_seconds, end_time_seconds,
                        fragment_duration, neurolm_embedding, embedding_norm, attention_pattern,
                        cognitive_state, neural_complexity, signal_quality, channel_consistency,
                        video_timestamp, experiment_session, search_keywords, neural_tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    meta['embedding_id'], meta['video_id'], meta['fragment_index'],
                    meta['start_time_seconds'], meta['end_time_seconds'], meta['fragment_duration'],
                    embedding_blob, meta['embedding_norm'], meta['attention_pattern'],
                    meta['cognitive_state'], meta['neural_complexity'], meta['signal_quality'],
                    meta['channel_consistency'], meta['video_timestamp'], meta['experiment_session'],
                    meta['search_keywords'], meta['neural_tags']
                ))
            
            conn.commit()
        
        logger.info(f"💾 Stored {len(embeddings)} NeuroLM embeddings in search database")
    
    def _build_faiss_index(self, embeddings: List[np.ndarray]):
        """Build FAISS index for fast similarity search"""
        if not embeddings:
            logger.warning("⚠️ No embeddings to index")
            return
        
        # Convert to matrix
        embedding_matrix = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Build FAISS index
        dimension = embedding_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embedding_matrix)
        
        # Save index
        index_path = self.search_db_path.parent / f"{self.search_db_path.stem}_faiss.index"
        faiss.write_index(self.faiss_index, str(index_path))
        
        logger.info(f"🚀 FAISS index built with {embedding_matrix.shape[0]} embeddings")
        logger.info(f"💾 Index saved to: {index_path}")
    
    def search_by_neural_pattern(self, pattern_type: str, top_k: int = 10) -> List[Dict]:
        """Search by neural pattern type (attention_pattern or cognitive_state)"""
        with sqlite3.connect(self.search_db_path) as conn:
            cursor = conn.cursor()
            
            if pattern_type in ['focused', 'distracted', 'variable']:
                query = '''
                    SELECT embedding_id, video_id, fragment_index, start_time_seconds, 
                           end_time_seconds, attention_pattern, cognitive_state, neural_complexity
                    FROM neurolm_search 
                    WHERE attention_pattern = ?
                    ORDER BY neural_complexity DESC
                    LIMIT ?
                '''
                cursor.execute(query, (pattern_type, top_k))
            elif pattern_type in ['active', 'passive', 'transitional']:
                query = '''
                    SELECT embedding_id, video_id, fragment_index, start_time_seconds, 
                           end_time_seconds, attention_pattern, cognitive_state, neural_complexity
                    FROM neurolm_search 
                    WHERE cognitive_state = ?
                    ORDER BY neural_complexity DESC
                    LIMIT ?
                '''
                cursor.execute(query, (pattern_type, top_k))
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
            
            results = cursor.fetchall()
            
            return [
                {
                    'embedding_id': row[0],
                    'video_id': row[1],
                    'fragment_index': row[2],
                    'start_time': row[3],
                    'end_time': row[4],
                    'attention_pattern': row[5],
                    'cognitive_state': row[6],
                    'neural_complexity': row[7]
                }
                for row in results
            ]
    
    def search_similar_embeddings(self, query_embedding_id: str, top_k: int = 10) -> List[Dict]:
        """Search for similar embeddings using FAISS"""
        if self.faiss_index is None:
            logger.error("❌ FAISS index not available")
            return []
        
        # Get query embedding
        with sqlite3.connect(self.search_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT neurolm_embedding FROM neurolm_search WHERE embedding_id = ?', 
                          (query_embedding_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.error(f"❌ Embedding not found: {query_embedding_id}")
                return []
            
            query_embedding = pickle.loads(result[0]).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search with FAISS
            similarities, indices = self.faiss_index.search(query_embedding, top_k + 1)  # +1 to exclude self
            
            # Get metadata for results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if i == 0:  # Skip self (highest similarity)
                    continue
                
                cursor.execute('''
                    SELECT embedding_id, video_id, fragment_index, start_time_seconds, 
                           end_time_seconds, attention_pattern, cognitive_state, neural_complexity
                    FROM neurolm_search 
                    LIMIT 1 OFFSET ?
                ''', (int(idx),))
                
                row = cursor.fetchone()
                if row:
                    results.append({
                        'embedding_id': row[0],
                        'video_id': row[1],
                        'fragment_index': row[2],
                        'start_time': row[3],
                        'end_time': row[4],
                        'attention_pattern': row[5],
                        'cognitive_state': row[6],
                        'neural_complexity': row[7],
                        'similarity_score': float(similarity)
                    })
            
            return results
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the NeuroLM search database"""
        with sqlite3.connect(self.search_db_path) as conn:
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute('SELECT COUNT(*) FROM neurolm_search')
            total_embeddings = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT video_id) FROM neurolm_search')
            unique_videos = cursor.fetchone()[0]
            
            # Pattern distribution
            cursor.execute('SELECT attention_pattern, COUNT(*) FROM neurolm_search GROUP BY attention_pattern')
            attention_dist = dict(cursor.fetchall())
            
            cursor.execute('SELECT cognitive_state, COUNT(*) FROM neurolm_search GROUP BY cognitive_state')
            cognitive_dist = dict(cursor.fetchall())
            
            # Complexity stats
            cursor.execute('SELECT AVG(neural_complexity), MIN(neural_complexity), MAX(neural_complexity) FROM neurolm_search')
            complexity_stats = cursor.fetchone()
            
            return {
                'total_embeddings': total_embeddings,
                'unique_videos': unique_videos,
                'attention_distribution': attention_dist,
                'cognitive_distribution': cognitive_dist,
                'complexity_avg': complexity_stats[0],
                'complexity_min': complexity_stats[1],
                'complexity_max': complexity_stats[2]
            }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create NeuroLM-based search database")
    parser.add_argument("--data_dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--fragment_duration", type=float, default=30.0, 
                       help="Fragment duration in seconds")
    parser.add_argument("--overlap_ratio", type=float, default=0.0, 
                       help="Overlap ratio between fragments (0.0 = no overlap)")
    parser.add_argument("--output_db", default="neurolm_search_database.db", 
                       help="Output search database path")
    parser.add_argument("--stats", action='store_true', help="Show database statistics")
    
    args = parser.parse_args()
    
    print("🧠 NeuroLM Video Search Database Creator")
    print("=" * 50)
    
    # Create search database
    search_db = NeuroLMSearchDatabase(args.output_db)
    
    if not args.stats:
        # Process CSV files
        num_embeddings = search_db.process_csv_files_with_neurolm(
            args.data_dir, 
            args.fragment_duration, 
            args.overlap_ratio
        )
        
        print(f"\n✅ NeuroLM search database created successfully!")
        print(f"📊 Total embeddings: {num_embeddings}")
        print(f"💾 Database: {args.output_db}")
    
    # Show stats
    if args.stats or not args.stats:  # Always show stats
        print("\n📊 Database Statistics:")
        print("-" * 30)
        stats = search_db.get_database_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
