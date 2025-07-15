from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Use MS MARCO or STS-based model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, docs: list[Document], top_k: int = 5):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:top_k]]
