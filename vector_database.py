#!/usr/bin/env python3
"""
NeuroLM Video Embedding Vector Database
Stores and searches video embeddings for similarity-based retrieval
"""

import numpy as np
import json
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # For efficient similarity search
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoEmbeddingVectorDB:
    """Vector database for storing and searching video embeddings"""
    
    def __init__(self, db_path: str = "video_embeddings.db", faiss_index_path: str = "video_embeddings.index"):
        self.db_path = Path(db_path)
        self.faiss_index_path = Path(faiss_index_path)
        self.embedding_dim = 768  # NeuroLM-B embedding dimension
        
        # Initialize SQLite database for metadata
        self.init_database()
        
        # Initialize FAISS index for fast similarity search
        self.faiss_index = None
        self.video_ids = []  # Maps FAISS index positions to video IDs
        self.load_or_create_faiss_index()
        
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
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_video_id ON video_embeddings(video_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_experiment_date ON video_embeddings(experiment_date)
            ''')
            
            conn.commit()
            logger.info("✅ Vector database initialized")
    
    def load_or_create_faiss_index(self):
        """Load existing FAISS index or create new one"""
        if self.faiss_index_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.faiss_index_path))
                
                # Load video ID mappings
                mapping_path = self.faiss_index_path.with_suffix('.mapping')
                if mapping_path.exists():
                    with open(mapping_path, 'r') as f:
                        self.video_ids = json.load(f)
                
                logger.info(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} embeddings")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load FAISS index: {e}")
                self.create_new_faiss_index()
        else:
            self.create_new_faiss_index()
    
    def create_new_faiss_index(self):
        """Create new FAISS index for similarity search"""
        # Use L2 distance for similarity (can be changed to inner product)
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.video_ids = []
        logger.info("✅ Created new FAISS index")
    
    def save_faiss_index(self):
        """Save FAISS index and mappings to disk"""
        try:
            faiss.write_index(self.faiss_index, str(self.faiss_index_path))
            
            # Save video ID mappings
            mapping_path = self.faiss_index_path.with_suffix('.mapping')
            with open(mapping_path, 'w') as f:
                json.dump(self.video_ids, f)
            
            logger.info(f"✅ Saved FAISS index with {self.faiss_index.ntotal} embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index: {e}")
    
    def store_video_embedding(self, 
                            video_id: str,
                            embedding: np.ndarray,
                            video_url: str = None,
                            video_title: str = None,
                            eeg_metrics: Dict = None,
                            experiment_date: str = None,
                            duration_seconds: float = None) -> bool:
        """Store video embedding and metadata in the database"""
        try:
            # Ensure embedding is the right shape
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim == 2 and embedding.shape[0] > 1:
                # Average multiple embeddings if provided
                embedding = np.mean(embedding, axis=0, keepdims=True)
            
            if embedding.shape[1] != self.embedding_dim:
                logger.error(f"❌ Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[1]}")
                return False
            
            # Calculate embedding statistics
            embedding_stats = {
                'mean': float(np.mean(embedding)),
                'std': float(np.std(embedding)),
                'min': float(np.min(embedding)),
                'max': float(np.max(embedding)),
                'norm': float(np.linalg.norm(embedding))
            }
            
            # Store in SQLite database
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
                    
                    # Update FAISS index
                    try:
                        idx = self.video_ids.index(video_id)
                        # Remove old embedding and add new one
                        self.faiss_index.remove_ids(np.array([idx]))
                        self.faiss_index.add(embedding.astype(np.float32))
                        logger.info(f"✅ Updated embedding for video: {video_id}")
                    except ValueError:
                        # Video not in FAISS index, add it
                        self.faiss_index.add(embedding.astype(np.float32))
                        self.video_ids.append(video_id)
                        logger.info(f"✅ Added new embedding for video: {video_id}")
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
                    
                    # Add to FAISS index
                    self.faiss_index.add(embedding.astype(np.float32))
                    self.video_ids.append(video_id)
                    logger.info(f"✅ Stored new embedding for video: {video_id}")
                
                conn.commit()
            
            # Save FAISS index
            self.save_faiss_index()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store video embedding: {e}")
            return False
    
    def search_similar_videos(self, 
                            query_embedding: np.ndarray, 
                            k: int = 5,
                            include_metadata: bool = True) -> List[Dict]:
        """Search for similar videos based on embedding similarity"""
        try:
            if self.faiss_index.ntotal == 0:
                logger.warning("⚠️ No embeddings in database")
                return []
            
            # Ensure query embedding is the right shape
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            elif query_embedding.ndim == 2 and query_embedding.shape[0] > 1:
                query_embedding = np.mean(query_embedding, axis=0, keepdims=True)
            
            if query_embedding.shape[1] != self.embedding_dim:
                logger.error(f"❌ Query embedding dimension mismatch: expected {self.embedding_dim}, got {query_embedding.shape[1]}")
                return []
            
            # Search using FAISS
            k = min(k, self.faiss_index.ntotal)  # Don't search for more than available
            distances, indices = self.faiss_index.search(query_embedding.astype(np.float32), k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.video_ids):
                    video_id = self.video_ids[idx]
                    similarity_score = 1.0 / (1.0 + distance)  # Convert L2 distance to similarity
                    
                    result = {
                        'video_id': video_id,
                        'similarity_score': float(similarity_score),
                        'distance': float(distance),
                        'rank': i + 1
                    }
                    
                    # Add metadata if requested
                    if include_metadata:
                        metadata = self.get_video_metadata(video_id)
                        if metadata:
                            result.update(metadata)
                    
                    results.append(result)
            
            logger.info(f"🔍 Found {len(results)} similar videos")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search similar videos: {e}")
            return []
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """Get metadata for a specific video"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT video_url, video_title, eeg_metrics, experiment_date, 
                           duration_seconds, embedding_stats, created_at
                    FROM video_embeddings WHERE video_id = ?
                ''', (video_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'video_url': row[0],
                        'video_title': row[1],
                        'eeg_metrics': json.loads(row[2]) if row[2] else None,
                        'experiment_date': row[3],
                        'duration_seconds': row[4],
                        'embedding_stats': json.loads(row[5]) if row[5] else None,
                        'created_at': row[6]
                    }
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get video metadata: {e}")
            return None
    
    def get_video_embedding(self, video_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific video"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT embedding_data FROM video_embeddings WHERE video_id = ?', (video_id,))
                row = cursor.fetchone()
                
                if row and row[0]:
                    return pickle.loads(row[0])
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get video embedding: {e}")
            return None
    
    def list_all_videos(self, limit: int = 100) -> List[Dict]:
        """List all videos in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT video_id, video_url, video_title, experiment_date, 
                           duration_seconds, created_at
                    FROM video_embeddings 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'video_id': row[0],
                        'video_url': row[1],
                        'video_title': row[2],
                        'experiment_date': row[3],
                        'duration_seconds': row[4],
                        'created_at': row[5]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Failed to list videos: {e}")
            return []
    
    def delete_video(self, video_id: str) -> bool:
        """Delete a video and its embedding from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM video_embeddings WHERE video_id = ?', (video_id,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    
                    # Remove from FAISS index (requires rebuilding)
                    if video_id in self.video_ids:
                        self.rebuild_faiss_index()
                    
                    logger.info(f"✅ Deleted video: {video_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Video not found: {video_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to delete video: {e}")
            return False
    
    def rebuild_faiss_index(self):
        """Rebuild FAISS index from database"""
        try:
            logger.info("🔄 Rebuilding FAISS index...")
            
            # Create new index
            self.create_new_faiss_index()
            
            # Load all embeddings from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT video_id, embedding_data FROM video_embeddings ORDER BY id')
                
                embeddings = []
                video_ids = []
                
                for video_id, embedding_data in cursor.fetchall():
                    if embedding_data:
                        embedding = pickle.loads(embedding_data)
                        embeddings.append(embedding.flatten())
                        video_ids.append(video_id)
                
                if embeddings:
                    embeddings_array = np.vstack(embeddings).astype(np.float32)
                    self.faiss_index.add(embeddings_array)
                    self.video_ids = video_ids
                    
                    self.save_faiss_index()
                    logger.info(f"✅ Rebuilt FAISS index with {len(embeddings)} embeddings")
                else:
                    logger.info("ℹ️ No embeddings to rebuild")
                    
        except Exception as e:
            logger.error(f"❌ Failed to rebuild FAISS index: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count total videos
                cursor.execute('SELECT COUNT(*) FROM video_embeddings')
                total_videos = cursor.fetchone()[0]
                
                # Get date range
                cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM video_embeddings')
                date_range = cursor.fetchone()
                
                # Get average duration
                cursor.execute('SELECT AVG(duration_seconds) FROM video_embeddings WHERE duration_seconds IS NOT NULL')
                avg_duration = cursor.fetchone()[0]
                
                return {
                    'total_videos': total_videos,
                    'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
                    'embedding_dimension': self.embedding_dim,
                    'earliest_video': date_range[0],
                    'latest_video': date_range[1],
                    'average_duration_seconds': avg_duration
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {}


def main():
    """Test the vector database functionality"""
    print("🧠 NeuroLM Video Embedding Vector Database")
    print("=" * 50)
    
    # Initialize database
    db = VideoEmbeddingVectorDB()
    
    # Get stats
    stats = db.get_database_stats()
    print(f"📊 Database Stats: {stats}")
    
    # List videos
    videos = db.list_all_videos(limit=10)
    print(f"📹 Recent Videos ({len(videos)}):")
    for video in videos:
        print(f"  - {video['video_id']}: {video['video_title']}")


if __name__ == "__main__":
    main()
