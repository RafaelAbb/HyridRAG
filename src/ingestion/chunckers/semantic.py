import re
from typing import Iterator

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.ingestion.base import ChunckerInterface, Chunk, ChunkingStrategy

TRANSOFRMER_MODEL = "all-MiniLM-L6-v2"
THRESHOLD_PERCENTILE = 25  # Adjust this value based on your needs

class SemanticChuncker(ChunckerInterface):
    
    def __init__(self, model_name=TRANSOFRMER_MODEL):
        super().__init__()
        self.model = SentenceTransformer(model_name)
    
    
    def chunk(self, doc, threshold=THRESHOLD_PERCENTILE) -> Iterator[Chunk]:

        metadata  = doc.metadata
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', doc.content) if s.strip()]

        if len(sentences) <= 1:
            yield Chunk(content=doc.content, metadata=metadata, chunk_id=0, chunk_strategy=ChunkingStrategy.SEMANTIC)
            return
            
        
        embeddings = self.model.encode(sentences) 

        
        scores = np.diag(cosine_similarity(embeddings[:-1], embeddings[1:]))  # Compute cosine similarity between adjacent sentences
        
        cutoff = np.percentile(scores, threshold)          # the value
        break_indices = np.where(scores < cutoff)[0]       # indices where similarity is low

        cuts = [0] + [i + 1 for i in break_indices] + [len(sentences)]
        for chunk_id, (start, end) in enumerate(zip(cuts, cuts[1:])):
            group = sentences[start:end]
            yield Chunk(content=" ".join(group), metadata=metadata, chunk_id=chunk_id, chunk_strategy=ChunkingStrategy.SEMANTIC)
