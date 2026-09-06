import os 
from pdf_rag_poc.parser import *
from pdf_rag_poc.chunker import SimpleChunker
from pdf_rag_poc.embedder import SentenceTransformerEmbedder
from pdf_rag_poc.llm_gateway import OllamaLLMHandler
from pdf_rag_poc.repository import ChromaDBRepository
from pdf_rag_poc.data_structure import Document

PDF_PATH = "./data/pdfs/"
IMG_PATH = "./data/imgs/"

PDF_FILE_1 = os.path.join(PDF_PATH, "tenant_alpha", "fictional_company_contract.pdf")
PDF_FILE_2 = os.path.join(PDF_PATH, "tenant_beta", "fictional_company_performance_report.pdf")

GENERATE_IMAGES = False

PROMPT = """ Generate answers to the following questions based only on the provided context. 
    If you think the information is not in context provided by the client, politely decline to answer. 
    Context:
    {context}
    Question:
    {question}
"""

def main():
    # Initilize Models
    chunker = SimpleChunker()
    embedder = SentenceTransformerEmbedder()
    llm_handler = OllamaLLMHandler()
    repository = ChromaDBRepository(embedder)

    # PDF Pre-Processing
    if GENERATE_IMAGES:
        dump_images(PDF_FILE_1, IMG_PATH)
        dump_images(PDF_FILE_2, IMG_PATH)

    imgs_alpha = load_images(os.path.join(IMG_PATH, "fictional_company_contract"))
    imgs_beta = load_images(os.path.join(IMG_PATH, "fictional_company_performance_report"))

    # Parsing
    info_tenant_alpha = parse_pdf_images(imgs_alpha)
    info_tenant_beta = parse_pdf_images(imgs_beta)

    # Chunking 
    doc_alpha_chunks = chunker.chunk(info_tenant_alpha)
    doc_beta_chunks = chunker.chunk(info_tenant_beta)

    # Embedding
    doc_alpha_embeddings = embedder.embed(doc_alpha_chunks)
    doc_beta_embeddings = embedder.embed(doc_beta_chunks)

    # Putting things together
    tenant_alpha_doc = Document(
        tenant_id="tenant_alpha",
        file_name="fictional_company_contract.pdf",
        text=info_tenant_alpha, 
        chunks=doc_alpha_embeddings, 
        )
    tenant_beta_doc = Document(
        tenant_id="tenant_beta",
        file_name="fictional_company_performance_report.pdf",
        text=info_tenant_beta, 
        chunks=doc_beta_embeddings, 
        )

    # Saving Embeddings
    repository.add([tenant_alpha_doc, tenant_beta_doc])

    # Querying and Retrieving
    query = "What is company name?"
    q_and_a = repository.query(query, "tenant_alpha")

    # Responding
    response = llm_handler.generate(question=q_and_a.question, context=q_and_a.answer, prompt=PROMPT)
    print("\nAnswer: ", response)

if __name__ == "__main__":
    main()
