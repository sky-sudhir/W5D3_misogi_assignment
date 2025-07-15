from langchain.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from chroma_client import get_vectorstore
from langchain_groq import ChatGroq

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

def get_compression_retriever():
    base_retriever = get_vectorstore().as_retriever(search_kwargs={"k": 8})
    
    compressor = LLMChainFilter.from_llm(llm)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
