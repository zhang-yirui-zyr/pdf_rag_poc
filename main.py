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
async def upload_file(file: Annotated[UploadFile, File()], tenant_id: Annotated[str, Form()], tenant_token: Annotated[str, Form()]):
    """
    Usage: 
        curl -X POST 'http://localhost:8000/upload/' \
             -F 'file=@data/pdfs/tenant_alpha/fictional_company_contract.pdf' \
             -F 'tenant_id="tenant_alpha"' \
             -F 'tenant_token="tenant_alpha_token"'
    """
    if not authenticate(tenant_id, tenant_token):
        raise HTTPException(status_code=401, detail=f"{tenant_id} with {tenant_token} is Unauthorized")
    else: 
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = file.filename
        
        return {"file": file_name, "tenant_id": tenant_id, "tenant_token": tenant_token}


@app.post("/question/")
async def read_item(query: FastAPIQuery):
    """
    Usage: 
        curl -X POST http://localhost:8000/question/ \
            -H "Content-Type: application/json" \
            -d '{"question": "What is the weather today?", "tenant_id": "tenant_alpha", "tenant_token": "tenant_alpha_token"}'
    """
    if authenticate(query.tenant_id, query.tenant_token):
        question: str = query.question
        return {"question": query.question, "tenant_id": query.tenant_id, "tenant_token": query.tenant_token}
    else:
        raise HTTPException(status_code=401, detail=f"{query.tenant_id} with {query.tenant_token} is Unauthorized")