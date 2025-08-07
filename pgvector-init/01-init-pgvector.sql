-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table for all data types
CREATE TABLE IF NOT EXISTS data_embeddings (
    id SERIAL PRIMARY KEY,
    data_type VARCHAR(50) NOT NULL, -- 'task', 'risk', 'quality', 'system', 'report'
    data_id INTEGER NOT NULL,
    content_text TEXT NOT NULL,
    embedding vector(1536), -- OpenAI compatible embedding dimension
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient vector search
CREATE INDEX IF NOT EXISTS idx_data_embeddings_type ON data_embeddings(data_type);
CREATE INDEX IF NOT EXISTS idx_data_embeddings_data_id ON data_embeddings(data_id);
CREATE INDEX IF NOT EXISTS idx_data_embeddings_vector ON data_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER update_data_embeddings_updated_at 
    BEFORE UPDATE ON data_embeddings 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create function for semantic search
CREATE OR REPLACE FUNCTION semantic_search(
    query_embedding vector(1536),
    data_type_filter VARCHAR(50) DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.7,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE(
    id INTEGER,
    data_type VARCHAR(50),
    data_id INTEGER,
    content_text TEXT,
    similarity FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        de.id,
        de.data_type,
        de.data_id,
        de.content_text,
        1 - (de.embedding <=> query_embedding) as similarity,
        de.metadata
    FROM data_embeddings de
    WHERE (data_type_filter IS NULL OR de.data_type = data_type_filter)
    AND 1 - (de.embedding <=> query_embedding) > similarity_threshold
    ORDER BY de.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get context for AI responses
CREATE OR REPLACE FUNCTION get_ai_context(
    query_text TEXT,
    data_types VARCHAR(50)[] DEFAULT ARRAY['task', 'risk', 'quality', 'system', 'report'],
    max_results INTEGER DEFAULT 5
)
RETURNS TABLE(
    context_text TEXT,
    data_type VARCHAR(50),
    data_id INTEGER,
    relevance_score FLOAT
) AS $$
DECLARE
    query_embedding vector(1536);
BEGIN
    -- Note: In production, you would generate the embedding here
    -- For now, we'll use a placeholder and return relevant data
    RETURN QUERY
    SELECT 
        de.content_text as context_text,
        de.data_type,
        de.data_id,
        0.8 as relevance_score
    FROM data_embeddings de
    WHERE de.data_type = ANY(data_types)
    AND de.content_text ILIKE '%' || query_text || '%'
    ORDER BY de.updated_at DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql; 