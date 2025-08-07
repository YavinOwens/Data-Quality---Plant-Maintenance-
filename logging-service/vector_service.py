import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from psycopg2.extras import RealDictCursor
import os

logger = logging.getLogger(__name__)

class VectorService:
    """Service for handling vector embeddings and semantic search using TF-IDF"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.vectorizer = None
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize the TF-IDF vectorizer"""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )
            logger.info("TF-IDF vectorizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TF-IDF vectorizer: {e}")
            self.vectorizer = None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate TF-IDF embedding for given text"""
        if not self.vectorizer or not text:
            return None
        
        try:
            # Fit and transform the text
            embedding = self.vectorizer.fit_transform([text])
            return embedding.toarray()[0].tolist()
        except Exception as e:
            logger.error(f"Failed to generate TF-IDF embedding: {e}")
            return None
    
    def store_embedding(self, data_type: str, data_id: int, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Store embedding in database"""
        try:
            embedding = self.generate_embedding(content)
            if not embedding:
                return False
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if embedding already exists
            cursor.execute(
                "SELECT id FROM data_embeddings WHERE data_type = %s AND data_id = %s",
                (data_type, data_id)
            )
            
            if cursor.fetchone():
                # Update existing embedding
                cursor.execute("""
                    UPDATE data_embeddings 
                    SET content_text = %s, embedding = %s, metadata = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE data_type = %s AND data_id = %s
                """, (content, embedding, json.dumps(metadata or {}), data_type, data_id))
            else:
                # Insert new embedding
                cursor.execute("""
                    INSERT INTO data_embeddings (data_type, data_id, content_text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (data_type, data_id, content, embedding, json.dumps(metadata or {})))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Stored TF-IDF embedding for {data_type} {data_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return False
    
    def semantic_search(self, query: str, data_types: List[str] = None, 
                       similarity_threshold: float = 0.1, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using TF-IDF cosine similarity"""
        try:
            # Get all embeddings for the specified data types
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query_sql = """
                SELECT id, data_type, data_id, content_text, embedding, metadata
                FROM data_embeddings
            """
            params = []
            
            if data_types:
                placeholders = ','.join(['%s'] * len(data_types))
                query_sql += f" WHERE data_type IN ({placeholders})"
                params.extend(data_types)
            
            cursor.execute(query_sql, params)
            results = cursor.fetchall()
            
            if not results:
                cursor.close()
                conn.close()
                return []
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                cursor.close()
                conn.close()
                return []
            
            # Calculate similarities
            similarities = []
            for row in results:
                try:
                    stored_embedding = row['embedding']
                    if stored_embedding:
                        # Calculate cosine similarity
                        similarity = cosine_similarity(
                            [query_embedding], 
                            [stored_embedding]
                        )[0][0]
                        
                        if similarity > similarity_threshold:
                            similarities.append({
                                'id': row['id'],
                                'data_type': row['data_type'],
                                'data_id': row['data_id'],
                                'content_text': row['content_text'],
                                'similarity': float(similarity),
                                'metadata': row['metadata']
                            })
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity for row {row['id']}: {e}")
                    continue
            
            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results = similarities[:limit]
            
            cursor.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    def get_ai_context(self, query: str, data_types: List[str] = None, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get relevant context for AI responses"""
        try:
            # First try semantic search
            semantic_results = self.semantic_search(query, data_types, limit=max_results)
            
            if semantic_results:
                return semantic_results
            
            # Fallback to text-based search
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query_sql = """
                SELECT 
                    id, data_type, data_id, content_text, 
                    0.8 as similarity, metadata
                FROM data_embeddings
                WHERE content_text ILIKE %s
            """
            params = [f'%{query}%']
            
            if data_types:
                placeholders = ','.join(['%s'] * len(data_types))
                query_sql += f" AND data_type IN ({placeholders})"
                params.extend(data_types)
            
            query_sql += " ORDER BY updated_at DESC LIMIT %s"
            params.append(max_results)
            
            cursor.execute(query_sql, params)
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get AI context: {e}")
            return []
    
    def populate_embeddings_from_data(self) -> Dict[str, int]:
        """Populate embeddings from existing data"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get all tasks
            cursor.execute("SELECT id, title, description, status FROM tasks")
            tasks = cursor.fetchall()
            
            # Get all risk assessments
            cursor.execute("SELECT id, risk_name, description, risk_level FROM risk_assessments")
            risks = cursor.fetchall()
            
            # Get all data quality records
            cursor.execute("SELECT id, metric_name, description, current_value FROM data_quality_metrics")
            quality_metrics = cursor.fetchall()
            
            # Get all system health records
            cursor.execute("SELECT id, component_name, status, description FROM system_health")
            system_health = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            counts = {
                'tasks': 0,
                'risks': 0,
                'quality': 0,
                'system': 0
            }
            
            # Store task embeddings
            for task_id, title, description, status in tasks:
                content = f"Task: {title}. Description: {description}. Status: {status}"
                metadata = {'title': title, 'status': status}
                if self.store_embedding('task', task_id, content, metadata):
                    counts['tasks'] += 1
            
            # Store risk embeddings
            for risk_id, risk_name, description, risk_level in risks:
                content = f"Risk: {risk_name}. Description: {description}. Level: {risk_level}"
                metadata = {'risk_name': risk_name, 'risk_level': risk_level}
                if self.store_embedding('risk', risk_id, content, metadata):
                    counts['risks'] += 1
            
            # Store quality metric embeddings
            for metric_id, metric_name, description, current_value in quality_metrics:
                content = f"Data Quality Metric: {metric_name}. Description: {description}. Value: {current_value}"
                metadata = {'metric_name': metric_name, 'current_value': current_value}
                if self.store_embedding('quality', metric_id, content, metadata):
                    counts['quality'] += 1
            
            # Store system health embeddings
            for health_id, component_name, status, description in system_health:
                content = f"System Component: {component_name}. Status: {status}. Description: {description}"
                metadata = {'component_name': component_name, 'status': status}
                if self.store_embedding('system', health_id, content, metadata):
                    counts['system'] += 1
            
            logger.info(f"Populated TF-IDF embeddings: {counts}")
            return counts
            
        except Exception as e:
            logger.error(f"Failed to populate embeddings: {e}")
            return {}
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    data_type,
                    COUNT(*) as count,
                    MIN(created_at) as oldest,
                    MAX(updated_at) as newest
                FROM data_embeddings 
                GROUP BY data_type
            """)
            
            results = cursor.fetchall()
            stats = {}
            
            for data_type, count, oldest, newest in results:
                stats[data_type] = {
                    'count': count,
                    'oldest': oldest.isoformat() if oldest else None,
                    'newest': newest.isoformat() if newest else None
                }
            
            cursor.close()
            conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {} 