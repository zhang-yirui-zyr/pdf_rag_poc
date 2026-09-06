from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
import os
import tempfile
from typing import Annotated
from contextlib import asynccontextmanager
from pdf_rag_poc import *
from pydantic import BaseModel
from joblib.externals.loky import get_reusable_executor
import json

class FastAPIQuery(BaseModel):
    question: str
    tenant_id: str
    tenant_token: str

def authenticate(tenant_id: str, tenant_token: str) -> bool:
    if tenant_id == "tenant_alpha" and tenant_token == "tenant_alpha_token":
        return True
    elif tenant_id == "tenant_beta" and tenant_token == "tenant_beta_token":
        return True
    else:
        return False

PROMPT = """ Generate answers to the following questions based only on the provided context. 
    If you think the information is not in context provided by the client, politely decline to answer. 
    Context:
    {context}
    Question:
    {question}
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initilize Models
    print("Starting up...")
    app.state.chunker = SimpleChunker()
    app.state.embedder = SentenceTransformerEmbedder()
    app.state.llm_handler = OllamaLLMHandler()
    app.state.repository = ChromaDBRepository(app.state.embedder)
    yield

    print("Shutting down...")
    del app.state.chunker
    del app.state.embedder
    del app.state.llm_handler
    del app.state.repository
    get_reusable_executor().shutdown(wait=True, kill_workers=True)

app = FastAPI(lifespan=lifespan)

@app.post("/upload/")
async def upload_file(
    file: Annotated[UploadFile, File()], 
    tenant_id: Annotated[str, Form()], 
    tenant_token: Annotated[str, Form()]):
    """
    Usage: 
        curl -X POST 'http://localhost:8000/upload/' \
             -F 'file=@data/pdfs/tenant_alpha/fictional_company_contract.pdf' \
             -F 'tenant_id="tenant_alpha"' \
             -F 'tenant_token="tenant_alpha_token"'
        curl -X POST 'http://localhost:8000/upload/' \
             -F 'file=@data/pdfs/tenant_beta/fictional_company_performance_report.pdf' \
             -F 'tenant_id="tenant_beta"' \
             -F 'tenant_token="tenant_beta_token"'
    Note: Currently, only one file can be uploaded, and one file should only be uploaded for once. 
    """
    if not authenticate(tenant_id, tenant_token):
        raise HTTPException(status_code=401, detail=f"{tenant_id} with {tenant_token} is Unauthorized")
    else: 
        with tempfile.TemporaryDirectory() as temp_dir:
            # Pre-Processing
            file_name = file.filename or "unknown.pdf"
            pdf_path = os.path.join(temp_dir, file_name)

            with open(pdf_path, "wb") as f:
                f.write(await file.read())

            dump_images(pdf_path, os.path.join(temp_dir, "imgs"))
            imgs = load_images(os.path.join(temp_dir, "imgs", file_name.split(".")[0]))

            # Parsing
            doc_text = parse_pdf_images(imgs)

            # Chunking
            doc_chunks = app.state.chunker.chunk(doc_text)

            # Embedding
            doc_embeddings = app.state.embedder.embed(doc_chunks)
            
            # Saving
            tenant_doc = Document(
                tenant_id=tenant_id,
                file_name=file_name,
                text=doc_text,
                chunks=doc_embeddings
            )
            app.state.repository.add([tenant_doc])
            return {"file": file_name, "num_chunks": len(doc_chunks), "saved_records": app.state.repository.collection.count()}

@app.post("/question/")
async def read_item(query: FastAPIQuery):
    """
    Usage: 
        curl -X POST http://localhost:8000/question/ \
            -H "Content-Type: application/json" \
            -d '{"question": "What is the weather today?", "tenant_id": "tenant_alpha", "tenant_token": "tenant_alpha_token"}'
            curl -X POST http://localhost:8000/question/ \
            -H "Content-Type: application/json" \
            -d '{"question": "What is the company name?", "tenant_id": "tenant_alpha", "tenant_token": "tenant_alpha_token"}'
            curl -X POST http://localhost:8000/question/ \
            -H "Content-Type: application/json" \
            -d '{"question": "What is the revenue?", "tenant_id": "tenant_alpha", "tenant_token": "tenant_alpha_token"}'
    Return:
        {
            "question":"What is the weather today?",
            "answer":"I'm not aware of any information about the weather provided in the context. 
            As there is no mention of weather, I cannot provide an answer to that question based on the given text."
        }
        {
            "question":"What is the company name?",
            "answer":"The company name is Northstar Dynamics Ltd."
        }
        {
            "question":"What is the revenue?",
            "answer":"The provided context does not specify the total revenue of Northstar Dynamics Ltd., 
            but it lists the fees for different roles and services:\n\n- Senior Consultant: €1,250 per day\n- 
            Consultant: €900 per day\n- Technical Specialist: €1,050 per day\n- Project Manager: €950 per day\n- 
            Training Workshop: €2,500 per session\n\nThese rates are exclusive of taxes, duties, or governmental charges."
        }
    """
    if not authenticate(query.tenant_id, query.tenant_token):
        raise HTTPException(status_code=401, detail=f"{query.tenant_id} with {query.tenant_token} is Unauthorized")
    else:
        # Retrieving
        q_and_a = app.state.repository.query(query.question, query.tenant_id)
        
        # Responding
        response = app.state.llm_handler.generate(question=q_and_a.question, context=q_and_a.answer, prompt=PROMPT)
        
        return {"question": query.question, "answer": response}