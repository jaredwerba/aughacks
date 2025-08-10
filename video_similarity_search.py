#!/usr/bin/env python3
"""
Video Similarity Search using Embeddings
Search for similar videos/fragments based on EEG embedding similarity
"""

import numpy as np
import sqlite3
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class VideoSimilaritySearcher:
    """Search for similar videos/fragments using embedding similarity"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.embeddings_cache = {}
        self.metadata_cache = {}
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load all embeddings and metadata from database"""
        print(f"📂 Loading embeddings from: {self.db_path}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if this is a fragment database or full video database
            cursor.execute("PRAGMA table_info(video_embeddings)")
            columns = [col[1] for col in cursor.fetchall()]
            is_fragment_db = 'fragment_id' in columns
            
            if is_fragment_db:
                query = '''
                    SELECT fragment_id, video_id, fragment_index, start_time_seconds, 
                           end_time_seconds, embedding_data, eeg_metrics, video_title,
                           fragment_duration_seconds, total_video_duration_seconds
                    FROM video_embeddings 
                    ORDER BY video_id, fragment_index
                '''
            else:
                query = '''
                    SELECT video_id, video_id as fragment_id, 0 as fragment_index, 
                           0 as start_time_seconds, duration_seconds as end_time_seconds,
                           embedding_data, eeg_metrics, video_title,
                           duration_seconds as fragment_duration_seconds, 
                           duration_seconds as total_video_duration_seconds
                    FROM video_embeddings 
                    ORDER BY video_id
                '''
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            embeddings_list = []
            ids_list = []
            
            for row in results:
                (fragment_id, video_id, fragment_index, start_time, end_time, 
                 embedding_data, eeg_metrics, video_title, fragment_duration, total_duration) = row
                
                # Load embedding
                embedding = pickle.loads(embedding_data)
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                embeddings_list.append(embedding)
                ids_list.append(fragment_id)
                
                # Store metadata
                self.metadata_cache[fragment_id] = {
                    'video_id': video_id,
                    'fragment_index': fragment_index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'video_title': video_title,
                    'fragment_duration': fragment_duration,
                    'total_duration': total_duration,
                    'eeg_metrics': json.loads(eeg_metrics) if eeg_metrics else {}
                }
            
            # Convert to numpy arrays
            self.embeddings_matrix = np.array(embeddings_list)
            self.embedding_ids = ids_list
            
            print(f"✅ Loaded {len(embeddings_list)} embeddings")
            print(f"📊 Embedding shape: {self.embeddings_matrix.shape}")
            print(f"🎬 Unique videos: {len(set(meta['video_id'] for meta in self.metadata_cache.values()))}")
    
    def find_similar_videos(self, query_id: str, top_k: int = 5, 
                           similarity_metric: str = 'cosine') -> List[Dict]:
        """
        Find similar videos/fragments to a query video/fragment
        
        Args:
            query_id: ID of the query video/fragment
            top_k: Number of similar items to return
            similarity_metric: 'cosine', 'euclidean', or 'manhattan'
        
        Returns:
            List of similar videos with similarity scores
        """
        if query_id not in self.embedding_ids:
            raise ValueError(f"Query ID '{query_id}' not found in database")
        
        # Get query embedding
        query_idx = self.embedding_ids.index(query_id)
        query_embedding = self.embeddings_matrix[query_idx].reshape(1, -1)
        
        # Calculate similarities
        if similarity_metric == 'cosine':
            similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
            # Convert to distance (higher = more similar)
            similarities = similarities
        elif similarity_metric == 'euclidean':
            distances = euclidean_distances(query_embedding, self.embeddings_matrix)[0]
            # Convert to similarity (lower distance = higher similarity)
            similarities = 1 / (1 + distances)
        elif similarity_metric == 'manhattan':
            distances = np.sum(np.abs(query_embedding - self.embeddings_matrix), axis=1)
            similarities = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        # Get top-k similar items (excluding the query itself)
        similar_indices = np.argsort(similarities)[::-1]
        
        results = []
        count = 0
        for idx in similar_indices:
            item_id = self.embedding_ids[idx]
            if item_id == query_id:  # Skip the query itself
                continue
            
            metadata = self.metadata_cache[item_id]
            result = {
                'id': item_id,
                'similarity_score': float(similarities[idx]),
                'video_id': metadata['video_id'],
                'fragment_index': metadata['fragment_index'],
                'start_time': metadata['start_time'],
                'end_time': metadata['end_time'],
                'duration': metadata['fragment_duration'],
                'video_title': metadata['video_title'],
                'eeg_metrics': metadata['eeg_metrics']
            }
            results.append(result)
            count += 1
            
            if count >= top_k:
                break
        
        return results
    
    def find_similar_by_video_id(self, video_id: str, top_k: int = 5, 
                                similarity_metric: str = 'cosine') -> List[Dict]:
        """Find videos similar to all fragments of a specific video"""
        # Get all fragments for this video
        video_fragments = [fid for fid in self.embedding_ids 
                          if self.metadata_cache[fid]['video_id'] == video_id]
        
        if not video_fragments:
            raise ValueError(f"Video ID '{video_id}' not found in database")
        
        print(f"🎬 Found {len(video_fragments)} fragments for video: {video_id}")
        
        # Calculate average similarity across all fragments
        all_similarities = {}
        
        for fragment_id in video_fragments:
            similar_items = self.find_similar_videos(fragment_id, top_k=50, 
                                                   similarity_metric=similarity_metric)
            
            for item in similar_items:
                other_video_id = item['video_id']
                if other_video_id == video_id:  # Skip same video
                    continue
                
                if other_video_id not in all_similarities:
                    all_similarities[other_video_id] = []
                all_similarities[other_video_id].append(item['similarity_score'])
        
        # Calculate average similarity per video
        video_similarities = []
        for other_video_id, scores in all_similarities.items():
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            
            # Get representative metadata
            representative_fragment = next(fid for fid in self.embedding_ids 
                                         if self.metadata_cache[fid]['video_id'] == other_video_id)
            metadata = self.metadata_cache[representative_fragment]
            
            video_similarities.append({
                'video_id': other_video_id,
                'avg_similarity': avg_score,
                'max_similarity': max_score,
                'num_comparisons': len(scores),
                'video_title': metadata['video_title'],
                'total_duration': metadata['total_duration']
            })
        
        # Sort by average similarity
        video_similarities.sort(key=lambda x: x['avg_similarity'], reverse=True)
        
        return video_similarities[:top_k]
    
    def get_embedding_statistics(self) -> Dict:
        """Get statistics about the embeddings"""
        stats = {
            'total_embeddings': len(self.embedding_ids),
            'embedding_dimension': self.embeddings_matrix.shape[1],
            'unique_videos': len(set(meta['video_id'] for meta in self.metadata_cache.values())),
            'embedding_mean': float(np.mean(self.embeddings_matrix)),
            'embedding_std': float(np.std(self.embeddings_matrix)),
            'embedding_min': float(np.min(self.embeddings_matrix)),
            'embedding_max': float(np.max(self.embeddings_matrix))
        }
        
        # Fragment statistics if applicable
        fragment_durations = [meta['fragment_duration'] for meta in self.metadata_cache.values()]
        if len(set(fragment_durations)) > 1:  # Multiple durations = fragment database
            stats['avg_fragment_duration'] = float(np.mean(fragment_durations))
            stats['total_duration'] = float(np.sum([meta['total_duration'] for meta in self.metadata_cache.values()]))
        
        return stats
    
    def visualize_similarity_matrix(self, video_ids: List[str] = None, save_path: str = None):
        """Create a similarity matrix visualization"""
        if video_ids is None:
            # Use all unique video IDs
            video_ids = list(set(meta['video_id'] for meta in self.metadata_cache.values()))[:10]  # Limit to 10 for readability
        
        print(f"📊 Creating similarity matrix for {len(video_ids)} videos...")
        
        # Get representative embeddings for each video (average of all fragments)
        video_embeddings = {}
        for video_id in video_ids:
            video_fragments = [self.embeddings_matrix[i] for i, fid in enumerate(self.embedding_ids) 
                             if self.metadata_cache[fid]['video_id'] == video_id]
            if video_fragments:
                video_embeddings[video_id] = np.mean(video_fragments, axis=0)
        
        # Calculate similarity matrix
        embedding_list = list(video_embeddings.values())
        similarity_matrix = cosine_similarity(embedding_list)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=[vid.replace('lsl_stream_', '') for vid in video_ids],
                   yticklabels=[vid.replace('lsl_stream_', '') for vid in video_ids],
                   annot=True, cmap='viridis', fmt='.3f')
        plt.title('Video Similarity Matrix (Cosine Similarity)')
        plt.xlabel('Videos')
        plt.ylabel('Videos')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Similarity matrix saved to: {save_path}")
        else:
            plt.show()
    
    def list_available_videos(self) -> List[Dict]:
        """List all available videos in the database"""
        videos = {}
        
        for fragment_id, metadata in self.metadata_cache.items():
            video_id = metadata['video_id']
            if video_id not in videos:
                videos[video_id] = {
                    'video_id': video_id,
                    'video_title': metadata['video_title'],
                    'total_duration': metadata['total_duration'],
                    'fragments': []
                }
            
            videos[video_id]['fragments'].append({
                'fragment_id': fragment_id,
                'fragment_index': metadata['fragment_index'],
                'start_time': metadata['start_time'],
                'end_time': metadata['end_time'],
                'duration': metadata['fragment_duration']
            })
        
        # Sort fragments by index
        for video_data in videos.values():
            video_data['fragments'].sort(key=lambda x: x['fragment_index'])
            video_data['num_fragments'] = len(video_data['fragments'])
        
        return list(videos.values())

def main():
    """Main function for similarity search"""
    parser = argparse.ArgumentParser(description="Video similarity search using embeddings")
    parser.add_argument("--db_path", required=True, help="Path to the embeddings database")
    parser.add_argument("--query_id", help="ID of the query video/fragment")
    parser.add_argument("--query_video", help="Video ID to find similar videos")
    parser.add_argument("--top_k", type=int, default=5, help="Number of similar items to return")
    parser.add_argument("--metric", choices=['cosine', 'euclidean', 'manhattan'], 
                       default='cosine', help="Similarity metric")
    parser.add_argument("--list_videos", action='store_true', help="List all available videos")
    parser.add_argument("--stats", action='store_true', help="Show embedding statistics")
    parser.add_argument("--visualize", action='store_true', help="Create similarity matrix visualization")
    
    args = parser.parse_args()
    
    print("🔍 Video Similarity Search System")
    print("=" * 50)
    
    # Initialize searcher
    searcher = VideoSimilaritySearcher(args.db_path)
    
    if args.stats:
        print("\n📊 Embedding Statistics:")
        print("-" * 30)
        stats = searcher.get_embedding_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    if args.list_videos:
        print("\n📋 Available Videos:")
        print("-" * 30)
        videos = searcher.list_available_videos()
        for i, video in enumerate(videos, 1):
            print(f"{i}. 🎬 {video['video_id']}")
            print(f"   📊 Fragments: {video['num_fragments']}")
            print(f"   ⏱️ Duration: {video['total_duration']:.1f}s")
            if video['fragments']:
                print(f"   🔗 First fragment ID: {video['fragments'][0]['fragment_id']}")
            print()
    
    if args.query_id:
        print(f"\n🔍 Finding videos similar to fragment: {args.query_id}")
        print("-" * 50)
        try:
            similar_items = searcher.find_similar_videos(args.query_id, args.top_k, args.metric)
            
            for i, item in enumerate(similar_items, 1):
                print(f"{i}. 🎯 {item['id']}")
                print(f"   📊 Similarity: {item['similarity_score']:.4f}")
                print(f"   🎬 Video: {item['video_id']}")
                print(f"   ⏱️ Time: {item['start_time']:.1f}s - {item['end_time']:.1f}s")
                if 'attention' in item['eeg_metrics']:
                    attention = item['eeg_metrics']['attention']
                    print(f"   🧠 Focus: {attention.get('focus_level', 'N/A'):.3f}")
                print()
        except ValueError as e:
            print(f"❌ Error: {e}")
    
    if args.query_video:
        print(f"\n🔍 Finding videos similar to: {args.query_video}")
        print("-" * 50)
        try:
            similar_videos = searcher.find_similar_by_video_id(args.query_video, args.top_k, args.metric)
            
            for i, video in enumerate(similar_videos, 1):
                print(f"{i}. 🎯 {video['video_id']}")
                print(f"   📊 Avg Similarity: {video['avg_similarity']:.4f}")
                print(f"   📈 Max Similarity: {video['max_similarity']:.4f}")
                print(f"   🔢 Comparisons: {video['num_comparisons']}")
                print(f"   ⏱️ Duration: {video['total_duration']:.1f}s")
                print()
        except ValueError as e:
            print(f"❌ Error: {e}")
    
    if args.visualize:
        print("\n📊 Creating similarity matrix visualization...")
        searcher.visualize_similarity_matrix(save_path="similarity_matrix.png")

if __name__ == "__main__":
    main()
