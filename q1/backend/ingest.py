# ingest.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
import os
from chroma_client import get_vectorstore
from langchain_community.document_loaders.csv_loader import CSVLoader


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def ingest_document(content: str, metadata: dict = None):
    metadata = metadata or {}
    document = Document(page_content=content, metadata=metadata)
    chunks = text_splitter.split_documents([document])
    db = get_vectorstore()
    db.add_documents(chunks)
    return {"status": "success", "chunks_added": len(chunks)}

def ingest_pdf(file_path: str, metadata: dict = None):
    metadata = metadata or {}

    if file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    else:
        loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata.update(metadata)

    chunks = text_splitter.split_documents(documents)

    db = get_vectorstore()
    db.add_documents(chunks)
    return {"status": "success", "chunks_added": len(chunks)}
