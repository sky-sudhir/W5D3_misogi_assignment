import os
import uuid
import tempfile
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
from chroma_client import get_vectorstore
from langchain_community.vectorstores.utils import filter_complex_metadata

# Initialize vector store and embedding
# EMBEDDINGS = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
VECTORSTORE = get_vectorstore()

async def process_and_store_doc(file):
    contents = await file.read()
    file_ext = os.path.splitext(file.filename)[-1]
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{file_ext}")

    with open(temp_path, "wb") as f:
        f.write(contents)

    # Load using Docling
    loader = DoclingLoader(temp_path)
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    filtered_chunks = filter_complex_metadata(chunks)
    # Store in Chroma
    VECTORSTORE.add_documents(filtered_chunks)

    return {"status": "success", "chunks": len(chunks)}
