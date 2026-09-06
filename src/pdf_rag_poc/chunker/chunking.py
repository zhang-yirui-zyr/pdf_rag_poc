from abc import ABC, abstractmethod

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass

class SimpleChunker(Chunker):
    def chunk(self, text: str, chunk_size: int = 100, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            current_chunk = words[i:i+chunk_size]
            chunks.append(" ".join(current_chunk))
        return chunks

class SemanticChunker(Chunker):
    def chunk(self, text: str, max_chunk_size: int = 100, overlap: int = 50) -> list[str]:
        pass
