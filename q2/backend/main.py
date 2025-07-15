from fastapi import FastAPI, UploadFile, Form, File
from ingest import process_and_store_doc
from generator import generate_assessment
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

class AssessmentRequest(BaseModel):
    topic: str
    objectives: List[str]
    difficulty: str
    user_id: str

@app.post("/upload/")
async def upload_doc(file: UploadFile = File(...)):
    return await process_and_store_doc(file)

@app.post("/generate/")
async def generate(request: AssessmentRequest):
    return await generate_assessment(request)




if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    ) 