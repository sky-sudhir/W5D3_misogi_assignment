# main.py

from fastapi import FastAPI, UploadFile, File, Form
from ingest import ingest_document, ingest_pdf
from graph import run_graph_pipeline
from pydantic import BaseModel
import shutil
import os
from uuid import uuid4
import uvicorn

app = FastAPI()

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryInput(BaseModel):
    query: str

class IngestInput(BaseModel):
    content: str
    metadata: dict = {}

@app.post("/ask")
async def ask_query(input: QueryInput):
    result = await run_graph_pipeline(input.query)
    return {"response": result}

@app.post("/ingest")
async def upload_document(input: IngestInput):
    result = ingest_document(input.content, input.metadata)
    return result

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), source: str = Form(...), date: str = Form(...)):
    temp_filename = f"{uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    metadata = {"source": source, "date": date}
    result = ingest_pdf(file_path, metadata)

    os.remove(file_path)  # Optional: delete after ingestion
    return result



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    ) 