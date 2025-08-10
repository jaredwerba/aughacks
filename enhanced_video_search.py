#!/usr/bin/env python3
"""
Enhanced Video Search System
Advanced search capabilities for EEG video embeddings with semantic queries
"""

import numpy as np
import sqlite3
import pickle
import json
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedVideoSearchEngine:
    """Enhanced search engine for EEG video embeddings with semantic capabilities"""
    
    def __init__(self, db_paths: List[str]):
        self.db_paths = [Path(path) for path in db_paths]
        self.databases = {}
        self.faiss_indices = {}
        self.neural_clusters = {}
        self._load_all_databases()
        self._build_search_indices()
        self._analyze_neural_clusters()
    
    def _load_all_databases(self):
        """Load all embedding databases"""
        for db_path in self.db_paths:
            if not db_path.exists():
                logger.warning(f"⚠️ Database not found: {db_path}")
                continue
                
            db_name = db_path.stem
            logger.info(f"📂 Loading: {db_name}")
            
            embeddings = []
            metadata = []
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check schema
                cursor.execute("PRAGMA table_info(video_embeddings)")
                columns = [col[1] for col in cursor.fetchall()]
                is_fragment_db = 'fragment_id' in columns
                
                if is_fragment_db:
                    query = '''
                        SELECT fragment_id, video_id, fragment_index, start_time_seconds, 
                               end_time_seconds, embedding_data, eeg_metrics
                        FROM video_embeddings ORDER BY video_id, fragment_index
                    '''
                else:
                    query = '''
                        SELECT video_id, video_id, 0, 0, duration_seconds, embedding_data, eeg_metrics
                        FROM video_embeddings ORDER BY video_id
                    '''
                
                cursor.execute(query)
                for row in cursor.fetchall():
                    fragment_id, video_id, fragment_idx, start_time, end_time, embedding_data, eeg_metrics = row
                    
                    embedding = pickle.loads(embedding_data)
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    embeddings.append(embedding)
                    metadata.append({
                        'fragment_id': fragment_id,
                        'video_id': video_id,
                        'fragment_index': fragment_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'database': db_name,
                        'eeg_metrics': json.loads(eeg_metrics) if eeg_metrics else {}
                    })
            
            self.databases[db_name] = {
                'embeddings': np.array(embeddings),
                'metadata': metadata
            }
            
            logger.info(f"✅ Loaded {len(embeddings)} embeddings from {db_name}")
    
    def _build_search_indices(self):
        """Build FAISS indices for fast similarity search"""
        for db_name, db_data in self.databases.items():
            embeddings = db_data['embeddings'].astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Build index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
            
            self.faiss_indices[db_name] = index
            logger.info(f"🚀 Built FAISS index for {db_name}: {embeddings.shape[0]} embeddings")
    
    def _analyze_neural_clusters(self):
        """Analyze neural clusters in each database"""
        for db_name, db_data in self.databases.items():
            embeddings = db_data['embeddings']
            metadata = db_data['metadata']
            
            # Cluster embeddings to find neural patterns
            n_clusters = min(5, len(embeddings) // 10)  # Adaptive cluster count
            if n_clusters < 2:
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Analyze clusters
            clusters = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_embeddings = embeddings[cluster_mask]
                cluster_metadata = [metadata[j] for j in range(len(metadata)) if cluster_mask[j]]
                
                # Analyze cluster characteristics
                cluster_center = kmeans.cluster_centers_[i]
                cluster_videos = list(set(meta['video_id'] for meta in cluster_metadata))
                
                # Calculate cluster coherence
                if len(cluster_embeddings) > 1:
                    coherence = np.mean(cosine_similarity(cluster_embeddings))
                else:
                    coherence = 1.0
                
                clusters[f"cluster_{i}"] = {
                    'center': cluster_center,
                    'size': len(cluster_embeddings),
                    'videos': cluster_videos,
                    'coherence': coherence,
                    'pattern_type': self._classify_cluster_pattern(cluster_center, cluster_metadata)
                }
            
            self.neural_clusters[db_name] = clusters
            logger.info(f"🧠 Identified {n_clusters} neural clusters in {db_name}")
    
    def _classify_cluster_pattern(self, cluster_center: np.ndarray, metadata: List[Dict]) -> str:
        """Classify the type of neural pattern in a cluster"""
        # Analyze embedding characteristics
        center_mean = np.mean(cluster_center)
        center_std = np.std(cluster_center)
        center_max = np.max(cluster_center)
        
        # Analyze temporal distribution
        start_times = [meta['start_time'] for meta in metadata]
        time_variance = np.var(start_times) if len(start_times) > 1 else 0
        
        # Classify pattern
        if center_std > 0.5 and center_max > 1.0:
            return "high_activation"
        elif center_std < 0.2:
            return "stable_baseline"
        elif time_variance > 100:  # Scattered across time
            return "variable_response"
        else:
            return "sustained_pattern"
    
    def search_by_video_similarity(self, query_video_id: str, database: str = "video_embeddings_5s", 
                                  top_k: int = 5) -> List[Dict]:
        """Search for videos similar to a query video"""
        if database not in self.databases:
            raise ValueError(f"Database {database} not found")
        
        db_data = self.databases[database]
        embeddings = db_data['embeddings']
        metadata = db_data['metadata']
        
        # Get all fragments for the query video
        query_indices = [i for i, meta in enumerate(metadata) if meta['video_id'] == query_video_id]
        if not query_indices:
            raise ValueError(f"Video {query_video_id} not found in {database}")
        
        # Average embeddings for the query video
        query_embeddings = embeddings[query_indices]
        query_vector = np.mean(query_embeddings, axis=0).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search with FAISS
        similarities, indices = self.faiss_indices[database].search(query_vector, len(embeddings))
        
        # Group results by video and calculate average similarity
        video_similarities = {}
        for similarity, idx in zip(similarities[0], indices[0]):
            meta = metadata[idx]
            video_id = meta['video_id']
            
            if video_id == query_video_id:  # Skip self
                continue
                
            if video_id not in video_similarities:
                video_similarities[video_id] = []
            video_similarities[video_id].append(float(similarity))
        
        # Calculate average similarity per video
        results = []
        for video_id, sims in video_similarities.items():
            avg_sim = np.mean(sims)
            max_sim = np.max(sims)
            
            results.append({
                'video_id': video_id,
                'avg_similarity': avg_sim,
                'max_similarity': max_sim,
                'num_fragments': len(sims),
                'database': database
            })
        
        # Sort by average similarity
        results.sort(key=lambda x: x['avg_similarity'], reverse=True)
        return results[:top_k]
    
    def search_by_neural_cluster(self, database: str, cluster_id: str = None) -> Dict:
        """Search by neural cluster patterns"""
        if database not in self.neural_clusters:
            return {}
        
        clusters = self.neural_clusters[database]
        
        if cluster_id and cluster_id in clusters:
            return {cluster_id: clusters[cluster_id]}
        
        return clusters
    
    def search_by_temporal_pattern(self, database: str, time_range: Tuple[float, float] = None,
                                  pattern_type: str = None) -> List[Dict]:
        """Search by temporal patterns"""
        if database not in self.databases:
            return []
        
        metadata = self.databases[database]['metadata']
        results = []
        
        for i, meta in enumerate(metadata):
            # Filter by time range
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= meta['start_time'] <= end_time):
                    continue
            
            # Filter by pattern type (based on EEG metrics)
            if pattern_type:
                eeg_metrics = meta['eeg_metrics']
                attention = eeg_metrics.get('attention', {})
                focus = attention.get('focus_level', 0.5)
                
                if pattern_type == "high_focus" and focus < 0.7:
                    continue
                elif pattern_type == "low_focus" and focus > 0.3:
                    continue
                elif pattern_type == "medium_focus" and not (0.3 <= focus <= 0.7):
                    continue
            
            results.append({
                'fragment_id': meta['fragment_id'],
                'video_id': meta['video_id'],
                'start_time': meta['start_time'],
                'end_time': meta['end_time'],
                'database': database,
                'eeg_metrics': meta['eeg_metrics']
            })
        
        return results
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics across all databases"""
        stats = {
            'databases': {},
            'cross_database_analysis': {},
            'neural_patterns': {}
        }
        
        # Per-database stats
        for db_name, db_data in self.databases.items():
            embeddings = db_data['embeddings']
            metadata = db_data['metadata']
            
            video_ids = list(set(meta['video_id'] for meta in metadata))
            
            stats['databases'][db_name] = {
                'total_embeddings': len(embeddings),
                'unique_videos': len(video_ids),
                'embedding_dimension': embeddings.shape[1],
                'avg_embedding_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
                'fragment_duration': metadata[0].get('end_time', 0) - metadata[0].get('start_time', 0) if metadata else 0
            }
        
        # Cross-database analysis
        if len(self.databases) > 1:
            db_names = list(self.databases.keys())
            for i, db1 in enumerate(db_names):
                for db2 in db_names[i+1:]:
                    # Compare video-level similarities between databases
                    correlation = self._calculate_cross_db_correlation(db1, db2)
                    stats['cross_database_analysis'][f"{db1}_vs_{db2}"] = correlation
        
        # Neural pattern analysis
        for db_name, clusters in self.neural_clusters.items():
            stats['neural_patterns'][db_name] = {
                'num_clusters': len(clusters),
                'cluster_types': [cluster['pattern_type'] for cluster in clusters.values()],
                'avg_coherence': np.mean([cluster['coherence'] for cluster in clusters.values()]) if clusters else 0
            }
        
        return stats
    
    def _calculate_cross_db_correlation(self, db1: str, db2: str) -> float:
        """Calculate correlation between video similarities across databases"""
        # Get video-level embeddings for both databases
        videos1 = self._get_video_level_embeddings(db1)
        videos2 = self._get_video_level_embeddings(db2)
        
        # Find common videos
        common_videos = set(videos1.keys()) & set(videos2.keys())
        if len(common_videos) < 3:
            return 0.0
        
        # Calculate similarity matrices
        common_list = sorted(list(common_videos))
        emb1 = np.array([videos1[vid] for vid in common_list])
        emb2 = np.array([videos2[vid] for vid in common_list])
        
        sim1 = cosine_similarity(emb1)
        sim2 = cosine_similarity(emb2)
        
        # Calculate correlation between similarity matrices
        sim1_flat = sim1[np.triu_indices_from(sim1, k=1)]
        sim2_flat = sim2[np.triu_indices_from(sim2, k=1)]
        
        correlation = np.corrcoef(sim1_flat, sim2_flat)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _get_video_level_embeddings(self, database: str) -> Dict[str, np.ndarray]:
        """Get average embedding per video for a database"""
        if database not in self.databases:
            return {}
        
        db_data = self.databases[database]
        embeddings = db_data['embeddings']
        metadata = db_data['metadata']
        
        video_embeddings = {}
        for video_id in set(meta['video_id'] for meta in metadata):
            video_indices = [i for i, meta in enumerate(metadata) if meta['video_id'] == video_id]
            video_embs = embeddings[video_indices]
            video_embeddings[video_id] = np.mean(video_embs, axis=0)
        
        return video_embeddings

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced video search system")
    parser.add_argument("--db_paths", nargs='+', required=True, 
                       help="Paths to embedding databases")
    parser.add_argument("--query_video", help="Video ID to search for similar videos")
    parser.add_argument("--database", default="video_embeddings_5s", 
                       help="Database to search in")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--stats", action='store_true', help="Show comprehensive statistics")
    parser.add_argument("--clusters", action='store_true', help="Show neural cluster analysis")
    
    args = parser.parse_args()
    
    print("🔍 Enhanced EEG Video Search System")
    print("=" * 50)
    
    # Initialize search engine
    search_engine = EnhancedVideoSearchEngine(args.db_paths)
    
    if args.stats:
        print("\n📊 Comprehensive Database Statistics:")
        print("-" * 50)
        stats = search_engine.get_comprehensive_stats()
        
        print("\n🗄️ Database Information:")
        for db_name, db_stats in stats['databases'].items():
            duration = db_stats['fragment_duration']
            if duration == 0:
                duration_str = "Full Video"
            else:
                duration_str = f"{duration:.0f}s Fragments"
            
            print(f"\n📊 {db_name}:")
            print(f"   🎯 Type: {duration_str}")
            print(f"   📈 Embeddings: {db_stats['total_embeddings']}")
            print(f"   🎬 Videos: {db_stats['unique_videos']}")
            print(f"   📏 Dimensions: {db_stats['embedding_dimension']}")
            print(f"   🔢 Avg Norm: {db_stats['avg_embedding_norm']:.3f}")
        
        if stats['cross_database_analysis']:
            print("\n🔗 Cross-Database Correlations:")
            for comparison, correlation in stats['cross_database_analysis'].items():
                print(f"   {comparison}: {correlation:.3f}")
        
        if stats['neural_patterns']:
            print("\n🧠 Neural Pattern Analysis:")
            for db_name, patterns in stats['neural_patterns'].items():
                print(f"\n🎯 {db_name}:")
                print(f"   🔍 Clusters: {patterns['num_clusters']}")
                print(f"   🎨 Types: {', '.join(patterns['cluster_types'])}")
                print(f"   📊 Coherence: {patterns['avg_coherence']:.3f}")
    
    if args.clusters:
        print("\n🧠 Neural Cluster Analysis:")
        print("-" * 50)
        for db_name in search_engine.neural_clusters:
            clusters = search_engine.search_by_neural_cluster(db_name)
            print(f"\n📊 {db_name}:")
            for cluster_id, cluster_info in clusters.items():
                print(f"   🎯 {cluster_id}:")
                print(f"      📈 Size: {cluster_info['size']} fragments")
                print(f"      🎬 Videos: {len(cluster_info['videos'])}")
                print(f"      🔗 Coherence: {cluster_info['coherence']:.3f}")
                print(f"      🧠 Pattern: {cluster_info['pattern_type']}")
    
    if args.query_video:
        print(f"\n🔍 Searching for videos similar to: {args.query_video}")
        print(f"📊 Using database: {args.database}")
        print("-" * 50)
        
        try:
            results = search_engine.search_by_video_similarity(
                args.query_video, args.database, args.top_k
            )
            
            for i, result in enumerate(results, 1):
                print(f"{i}. 🎯 {result['video_id']}")
                print(f"   📊 Avg Similarity: {result['avg_similarity']:.4f}")
                print(f"   📈 Max Similarity: {result['max_similarity']:.4f}")
                print(f"   🔢 Fragments: {result['num_fragments']}")
                print()
        
        except ValueError as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
