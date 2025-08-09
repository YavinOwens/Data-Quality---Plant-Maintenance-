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
        """Legacy single-text TF-IDF embedding (not used for storage)."""
        if not self.vectorizer or not text:
            return None
        try:
            embedding = self.vectorizer.fit_transform([text])
            return embedding.toarray()[0].tolist()
        except Exception as e:
            logger.error(f"Failed to generate TF-IDF embedding: {e}")
            return None
    
    def store_embedding(self, data_type: str, data_id: int, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Store content and metadata; keep embedding NULL (computed at query time)."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            # Check if record already exists
            cursor.execute(
                "SELECT id FROM data_embeddings WHERE data_type = %s AND data_id = %s",
                (data_type, data_id)
            )
            if cursor.fetchone():
                cursor.execute(
                    """
                    UPDATE data_embeddings 
                    SET content_text = %s, embedding = NULL, metadata = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE data_type = %s AND data_id = %s
                    """,
                    (content, json.dumps(metadata or {}), data_type, data_id),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO data_embeddings (data_type, data_id, content_text, embedding, metadata)
                    VALUES (%s, %s, %s, NULL, %s)
                    """,
                    (data_type, data_id, content, json.dumps(metadata or {})),
                )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Stored embedding record for {data_type} {data_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return False
    
    def semantic_search(self, query: str, data_types: List[str] = None,
                        similarity_threshold: float = 0.1, limit: int = 10) -> List[Dict[str, Any]]:
        """Semantic search using TF-IDF over stored content_text (no persisted vectors)."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            sql = """
                SELECT id, data_type, data_id, content_text, metadata
                FROM data_embeddings
            """
            params: List[Any] = []
            if data_types:
                placeholders = ','.join(['%s'] * len(data_types))
                sql += f" WHERE data_type IN ({placeholders})"
                params.extend(data_types)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            if not rows:
                return []
            corpus = [row['content_text'] or '' for row in rows]
            # Fit TF-IDF on corpus + query
            vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
            )
            matrix = vectorizer.fit_transform(corpus + [query])
            corpus_matrix = matrix[:-1]
            query_vec = matrix[-1]
            sims = cosine_similarity(corpus_matrix, query_vec)
            results: List[Tuple[int, float]] = [(idx, float(sims[idx][0])) for idx in range(len(corpus))]
            results.sort(key=lambda x: x[1], reverse=True)
            output: List[Dict[str, Any]] = []
            for idx, score in results:
                if score < similarity_threshold:
                    break
                row = rows[idx]
                output.append({
                    'id': row['id'],
                    'data_type': row['data_type'],
                    'data_id': row['data_id'],
                    'content_text': row['content_text'],
                    'similarity': score,
                    'metadata': row['metadata'],
                })
                if len(output) >= limit:
                    break
            return output
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
        """Populate embeddings from actual Postgres tables available in the schema."""
        counts: Dict[str, int] = {
            'quality': 0,
            'validation': 0,
            'risks': 0,
            'security': 0,
            'audit': 0,
            'workflow': 0,
            'notification': 0,
            'config': 0,
            'issue': 0,
        }
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        except Exception as e:
            logger.error(f"Failed to connect to database for embedding population: {e}")
            return {}

        def safe_exec(sql: str, params: Tuple = ()):  # helper
            try:
                cursor.execute(sql, params)
                return cursor.fetchall()
            except Exception as ex:
                logger.warning(f"Populate skip due to query error: {ex}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                return []

        # data_quality_metrics
        for row in safe_exec("""
            SELECT id,
                   COALESCE(metric_name,'') AS metric_name,
                   COALESCE(metric_value::text,'') AS metric_value,
                   COALESCE(metric_unit,'') AS metric_unit,
                   COALESCE(status,'') AS status
            FROM data_quality_metrics
        """):
            content = (
                f"Data Quality Metric: {row['metric_name']} = {row['metric_value']} {row['metric_unit']} (status={row['status']})"
            )
            meta = {
                'metric_name': row['metric_name'],
                'metric_value': row['metric_value'],
                'metric_unit': row['metric_unit'],
                'status': row['status'],
            }
            if self.store_embedding('quality', int(row['id']), content, meta):
                counts['quality'] += 1

        # validation_results
        for row in safe_exec("""
            SELECT id, validation_type, validation_name,
                   COALESCE(success_rate::text,'') AS success_rate,
                   COALESCE(passed_records,0) AS passed_records,
                   COALESCE(failed_records,0) AS failed_records
            FROM validation_results
        """):
            content = (
                f"Validation: {row['validation_name']} ({row['validation_type']}). "
                f"Success Rate: {row['success_rate']}. Passed: {row['passed_records']} Failed: {row['failed_records']}"
            )
            meta = {
                'validation_type': row['validation_type'],
                'validation_name': row['validation_name'],
            }
            if self.store_embedding('validation', int(row['id']), content, meta):
                counts['validation'] += 1

        # security_events
        for row in safe_exec("""
            SELECT id, COALESCE(event_type,'') AS event_type,
                   COALESCE(severity,'') AS severity,
                   COALESCE(source_ip,'') AS source_ip,
                   COALESCE(username,'') AS username,
                   COALESCE(details,'') AS details
            FROM security_events
        """):
            content = (
                f"Security Event: {row['event_type']} Severity: {row['severity']} "
                f"User: {row['username']} Source: {row['source_ip']} Details: {row['details']}"
            )
            meta = {
                'event_type': row['event_type'],
                'severity': row['severity'],
                'username': row['username'],
            }
            if self.store_embedding('security', int(row['id']), content, meta):
                counts['security'] += 1

        # audit_logs (prefer plural table if present)
        for row in safe_exec("""
            SELECT id, COALESCE(username,'') AS username,
                   COALESCE(action,'') AS action,
                   COALESCE(resource,'') AS resource,
                   COALESCE(ip_address,'') AS ip_address,
                   COALESCE(details,'') AS details,
                   COALESCE(success, true) AS success
            FROM audit_logs
        """):
            content = (
                f"Audit: user={row['username']} action={row['action']} resource={row['resource']} "
                f"ip={row['ip_address']} success={row['success']} details={row['details']}"
            )
            meta = {
                'username': row['username'],
                'action': row['action'],
                'resource': row['resource'],
            }
            if self.store_embedding('audit', int(row['id']), content, meta):
                counts['audit'] += 1

        # workflow_execution_logs
        for row in safe_exec("""
            SELECT id, COALESCE(workflow_name,'') AS workflow_name,
                   COALESCE(status,'') AS status,
                   COALESCE(error_message,'') AS error_message
            FROM workflow_execution_logs
        """):
            content = f"Workflow Execution: {row['workflow_name']} status={row['status']} error={row['error_message']}"
            meta = {'status': row['status'], 'workflow_name': row['workflow_name']}
            if self.store_embedding('workflow', int(row['id']), content, meta):
                counts['workflow'] += 1

        # notifications
        for row in safe_exec("""
            SELECT id,
                   COALESCE(notification_type,'') AS notification_type,
                   COALESCE(subject,'') AS subject,
                   COALESCE(message,'') AS message
            FROM notifications
        """):
            content = f"Notification: [{row['notification_type']}] {row['subject']} - {row['message']}"
            meta = {'notification_type': row['notification_type'], 'subject': row['subject']}
            if self.store_embedding('notification', int(row['id']), content, meta):
                counts['notification'] += 1

        # configuration
        for row in safe_exec("""
            SELECT id, COALESCE(config_key,'') AS config_key, COALESCE(config_value,'') AS config_value
            FROM configuration
        """):
            content = f"Config: {row['config_key']}={row['config_value']}"
            meta = {'config_key': row['config_key']}
            if self.store_embedding('config', int(row['id']), content, meta):
                counts['config'] += 1

        # data_quality_issues
        for row in safe_exec("""
            SELECT id, COALESCE(issue_type,'') AS issue_type,
                   COALESCE(issue_description,'') AS issue_description,
                   COALESCE(severity,'') AS severity,
                   COALESCE(affected_records,0) AS affected_records
            FROM data_quality_issues
        """):
            content = (
                f"Issue: {row['issue_type']} Severity: {row['severity']} "
                f"Affected: {row['affected_records']} Desc: {row['issue_description']}"
            )
            meta = {
                'issue_type': row['issue_type'],
                'severity': row['severity'],
            }
            if self.store_embedding('issue', int(row['id']), content, meta):
                counts['issue'] += 1

        # risk_register
        for row in safe_exec("""
            SELECT id,
                   COALESCE(title,'') AS title,
                   COALESCE(category,'') AS category,
                   COALESCE(likelihood,0) AS likelihood,
                   COALESCE(impact,0) AS impact,
                   COALESCE(status,'') AS status,
                   COALESCE(owner,'') AS owner
            FROM risk_register
        """):
            content = (
                f"Risk: {row['title']} [{row['category']}] status={row['status']} "
                f"owner={row['owner']} score={(row['likelihood'] or 0)*(row['impact'] or 0)}"
            )
            meta = {
                'title': row['title'],
                'category': row['category'],
                'status': row['status'],
                'owner': row['owner'],
            }
            if self.store_embedding('risk', int(row['id']), content, meta):
                counts['risks'] += 1

        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

        logger.info(f"Populated content for embeddings: {counts}")
        return counts
    
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