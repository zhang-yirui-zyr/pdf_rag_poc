import requests
from dataclasses import dataclass
from pdf_rag_poc.repository import ChromaDBRepository
from pdf_rag_poc.data_structure import Document, Chunk
from pdf_rag_poc.embedder import SentenceTransformerEmbedder

# Initilaize Models
embedder = SentenceTransformerEmbedder()
repository = ChromaDBRepository(embedder)

# Prepare Test Data
paragraph: str = """Project Aurora is an internal software initiative launched in March 2026. Its goal is to automate the processing of customer support requests using a combination of retrieval and language models. The project is led by Maya Chen, with Daniel Weber responsible for infrastructure and deployment. The initial prototype uses Python, PostgreSQL, and a vector database for document retrieval. Aurora currently supports English-language documents and is designed to process up to 10,000 documents in its proof-of-concept environment. The team plans to evaluate retrieval accuracy in September 2026 before deciding whether to move toward production. The project's main limitation is that documents containing highly specialized technical terminology sometimes produce inaccurate or incomplete retrieval results."""
tenant_alpha_chunks: list[str] = ["AAA" + s + "AAA" for s in paragraph.split(".")]
tenant_beta_chunks: list[str] = ["BBB" + s + "BBB" for s in paragraph.split(".")]

test_tenant_alpha_doc = Document(
    tenant_id="tenant_alpha", 
    file_name="tenant_alpha.pdf", 
    text=paragraph, 
    chunks=embedder.embed(texts=tenant_alpha_chunks)
    )

test_tenant_beta_doc = Document(
    tenant_id="tenant_beta", 
    file_name="tenant_beta.pdf", 
    text=paragraph, 
    chunks=embedder.embed(texts=tenant_beta_chunks)
    )

repository.add([test_tenant_alpha_doc, test_tenant_beta_doc])

def test_multi_tenant_query():
    """
    Show that multi-tenant query do not return results that are in the documents of other tenants.
    """
    questions = ["What is the company name?", "What is the project name?", "What is the project description?"]
    for question in questions:
        q_and_a_alpha = repository.query(question, "tenant_alpha", top_n=3)
        q_and_a_beta = repository.query(question, "tenant_beta", top_n=3)
        assert q_and_a_alpha.answer != q_and_a_beta.answer
        for a in q_and_a_alpha.answer:
            assert "AAA" in a
            assert "BBB" not in a
        for b in q_and_a_beta.answer:
            assert "BBB" in b
            assert "AAA" not in b

