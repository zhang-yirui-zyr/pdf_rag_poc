from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from pdf_rag_poc.data_structure import Chunk

class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[Chunk]:
        pass

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)

    def embed(self, texts: list[str], batch_size: int = 1024) -> list[Chunk]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch_chunks = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_chunks)
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                c = Chunk(text=chunk, id=i, embedding=embedding)
                results.append(c)
        return results