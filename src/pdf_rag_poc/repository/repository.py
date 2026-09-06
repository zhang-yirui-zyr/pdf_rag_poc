from abc import ABC, abstractmethod
from typing import Any
from pdf_rag_poc.embedder import Embedder
from pdf_rag_poc.data_structure import Document, Chunk, QandContext
import chromadb

class Repository(ABC):
    @abstractmethod
    def add(self, docs: list[Any]) -> None:
        pass
    @abstractmethod
    def query(self, query: str, tenant_id: str, top_n: int) -> Any:
        pass

class ChromaDBRepository(Repository):
    def __init__(self, embedder: Embedder):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            "all_documents",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self.embedder = embedder
    
    def add(self, docs: list[Document]) -> None:
        for doc in docs:
            chunks: list[Chunk] = doc.chunks
            ids = [f"{doc.id}_{chunk.id}" for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            metadatas = [{"owner": doc.tenant_id, "doc_id": doc.id, "file_name": doc.file_name, "chunk_id": chunk.id} for chunk in chunks]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas, # type: ignore
            )

    def query(self, query: str, tenant_id: str, top_n: int = 3) -> QandContext:
        query_embedding = self.embedder.embed([query])[0].embedding
        response: chromadb.QueryResult = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            where={"owner": tenant_id},
        )
        qandc = QandContext(question=query, context=response["documents"][0]) # type: ignore
        return qandc
    
    def delete_all(self) -> None:
        self.client.delete_collection(self.collection.name)
