from pdf_rag_poc.repository import ChromaDBRepository
from pdf_rag_poc.data_structure import Document
from pdf_rag_poc.embedder import SentenceTransformerEmbedder

# Initilaize Models
embedder = SentenceTransformerEmbedder()
repository = ChromaDBRepository(embedder)

# Prepare Test Data
tenant_alpha_paragraph: str = """Northstar Analytics is a software company founded in 2018. The company produces softwares for business intelligence.\
Its platform provides dashboards, automated reports, and data visualization tools.\
Northstar Analytics has a small engineering team and primarily serves medium-sized businesses.\
The company is currently focused on improving the reliability and performance of its analytics platform.\
For customer support, users can contact the support team through the company's website. The company reviews its product roadmap on a quarterly basis."""

tenant_beta_paragraph: str = """Bluepeak Systems is a software company founded in 2021. The company produces softwares for cybersecurity.\
Its platform provides dashboards, automated reports, and data visualization tools.\
Bluepeak Systems has a big engineering team and primarily serves large corporations.\
The company is currently focused on improving the reliability and performance of its analytics platform.\
For customer support, users can contact the support team through the company's website. The company reviews its product roadmap on a quarterly basis.""" 

tenant_alpha_chunks: list[str] = [s for s in tenant_alpha_paragraph.split(".")]
tenant_beta_chunks: list[str] = [s for s in tenant_beta_paragraph.split(".")]

test_tenant_alpha_doc = Document(
    tenant_id="tenant_alpha", 
    file_name="tenant_alpha.pdf", 
    text=tenant_alpha_paragraph, 
    chunks=embedder.embed(texts=tenant_alpha_chunks)
    )

test_tenant_beta_doc = Document(
    tenant_id="tenant_beta", 
    file_name="tenant_beta.pdf", 
    text=tenant_beta_paragraph, 
    chunks=embedder.embed(texts=tenant_beta_chunks)
    )

repository.add([test_tenant_alpha_doc, test_tenant_beta_doc])

def test_multi_tenant_query():
    """
    Show that multi-tenant query do not return results that are in the documents of other tenants.
    """
    questions = ["What is the company name?", "What year is the company founded?", "What kind of software does the company produce?"]
    expected_answers_alpha = ["Northstar Analytics", "2018", "business intelligence"]
    expected_answers_beta = ["Bluepeak Systems", "2021", "cybersecurity"]

    alpha_contexts = []
    beta_contexts = []
    for question in questions:
        q_and_c_alpha = repository.query(question, "tenant_alpha", top_n=3)
        q_and_c_beta = repository.query(question, "tenant_beta", top_n=3)

        alpha_contexts.append(" ".join(q_and_c_alpha.context))
        beta_contexts.append(" ".join(q_and_c_beta.context))

    for a, c in zip(expected_answers_alpha, alpha_contexts):
        assert a in c
        print(f"Answer {a} in Context {c} for tenant alpha", a in c)
    for a, c in zip(expected_answers_beta, beta_contexts):
        assert a in c
        print(f"Answer {a} in Context {c} for tenant beta", a in c)
