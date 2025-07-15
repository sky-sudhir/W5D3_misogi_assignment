
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_vectorstore():
    embedding_fn = OllamaEmbeddings(model="nomic-embed-text")

    # Set the local persist directory
    persist_directory = os.getenv('CHROMA_PERSIST_DIR', "./chroma_db")

    # Create a local Chroma client with persistence
    client = chromadb.PersistentClient(path=persist_directory)

    return Chroma(
        client=client,
        collection_name="quiz_docs",
        embedding_function=embedding_fn
    )
