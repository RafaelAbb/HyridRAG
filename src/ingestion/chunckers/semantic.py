import re

import numpy as np
from sentence_transformers import SentenceTransformer
import sklearn
from src.ingestion.base import ChunckerInterface, Chunk
from src.ingestion.chuncker import ChunkingStrategy

TRANSOFRMER_MODEL = "all-MiniLM-L6-v2"
THRESHOLD_PERCENTILE = 25  # Adjust this value based on your needs

class SemanticChuncker(ChunckerInterface):
    
    def __init__(self, model_name=TRANSOFRMER_MODEL):
        super().__init__()
        self.model = SentenceTransformer(model_name)
    
    
    def chunk(self, doc, threshold=THRESHOLD_PERCENTILE) -> list[Chunk]:

        metadata  = doc.metadata
        sentences = re.split(r'(?<=[.!?])\s+', doc.content)
        
        embeddings = self.model.encode(sentences) 

        
        scores = np.diag(sklearn.metrics.pairwise.cosine_similarity(embeddings[:-1], embeddings[1:]))  # Compute cosine similarity between adjacent sentences
        
        cutoff = np.percentile(scores, threshold)          # the value
        break_indices = np.where(scores < cutoff)[0]       # indices where similarity is low

        cuts = [0] + [i + 1 for i in break_indices] + [len(sentences)]
        for chunk_id, (start, end) in enumerate(zip(cuts, cuts[1:])):
            group = sentences[start:end]
            yield Chunk(content=" ".join(group), metadata=metadata, chunk_id=chunk_id, chunk_strategy=ChunkingStrategy.SEMANTIC)
