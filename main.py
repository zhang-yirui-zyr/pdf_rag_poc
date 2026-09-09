from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import os
import tempfile
from typing import Annotated
from contextlib import asynccontextmanager
from pdf_rag_poc import *
from pydantic import BaseModel
from joblib.externals.loky import get_reusable_executor

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
    app.state.embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    # app.state.embedder = SentenceTransformerEmbedder(# For better embedding performance, try this embedder instead
    #     model_name="codefuse-ai/F2LLM-v2-0.6B", 
    #     model_kwargs={"torch_dtype": "bfloat16"})
    app.state.llm_handler = OllamaLLMHandler()
    app.state.repository = ChromaDBRepository(app.state.embedder)
    yield
    # Clean up
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
    tenant_token: Annotated[str, Form()]
    ):
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
            doc = Document(
                tenant_id=tenant_id,
                file_name=file_name,
                text=doc_text,
                chunks=doc_embeddings
            )
            app.state.repository.add([doc])
            return {"file": file_name, "num_chunks": len(doc_chunks), "saved_records": app.state.repository.collection.count()}

@app.post("/question/")
async def read_item(query: FastAPIQuery):
    if not authenticate(query.tenant_id, query.tenant_token):
        raise HTTPException(status_code=401, detail=f"{query.tenant_id} with {query.tenant_token} is Unauthorized")
    else:
        # Retrieving
        q_and_c = app.state.repository.query(query.question, query.tenant_id)
        
        # Responding
        response = app.state.llm_handler.generate(question=q_and_c.question, context=q_and_c.context, prompt=PROMPT)
        
        return {"question": query.question, "answer": response}