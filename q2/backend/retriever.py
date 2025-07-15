from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from chroma_client import get_vectorstore

# Load dense vector store (Chroma)
EMBEDDINGS = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
VECTORSTORE = get_vectorstore()

# Sparse Index Store (in-memory, could later persist using Redis or pickle)
BM25_INDEX = None
BM25_DOCS = None

def build_sparse_index():
    global BM25_INDEX, BM25_DOCS
    all_docs = VECTORSTORE.similarity_search("dummy", k=1000)  # Fetch large set
    corpus = [doc.page_content for doc in all_docs]
    BM25_INDEX = BM25Okapi([doc.split(" ") for doc in corpus])
    BM25_DOCS = all_docs
    print(f"Sparse index built with {len(corpus)} documents.")

def hybrid_retrieve(query: str, k=10) -> list[Document]:

    if BM25_INDEX is None:
        build_sparse_index()

    dense_results = VECTORSTORE.similarity_search(query, k=k)
    
    sparse_scores = BM25_INDEX.get_scores(query.split())
    
    # Combine: Score top k from BM25
    sparse_indices = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:k]
    sparse_results = [BM25_DOCS[i] for i in sparse_indices]

    combined = {doc.page_content: doc for doc in dense_results + sparse_results}
    return list(combined.values())[:k]
