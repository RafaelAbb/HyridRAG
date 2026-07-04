from pydantic_settings import BaseSettings

from src.ingestion.base import ChunkingStrategy


class Settings(BaseSettings):
    # LLM providers
    openai_api_key: str
    anthropic_api_key: str = ""

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "rag_docs"

    # Retrieval
    dense_top_k: int = 10
    sparse_top_k: int = 10
    reranker_top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    similarity_dedup_threshold: float = 0.95

    # Generation
    generation_model: str = "gpt-4o"
    generation_temperature: float = 0.0
    generation_max_tokens: int = 1024
    
    judgement_model: str = "gpt-4o-mini"
    judgement_temperature: float = 0.0
    judgement_max_tokens: int = 1024
    
    # Chunking
    default_chunk_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()