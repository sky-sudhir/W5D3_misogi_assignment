# graph.py

from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq as Groq
from langchain.embeddings import OllamaEmbeddings
from retreiver import get_compression_retriever
import numpy as np

# Groq LLM
llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# Shared retriever
compression_retriever = get_compression_retriever()


### ---------- 1. Graph State Definition ----------
class RAGState(Dict[str, Any]):
    query: str
    sub_queries: List[str]
    answers: List[Dict[str, Any]]


### ---------- 2. Query Decomposition ----------
decompose_prompt = PromptTemplate.from_template(
    "Decompose the following complex sports question into simpler sub-questions:\n\n{query}\n\nSub-questions:"
)
decompose_chain = (decompose_prompt | llm | StrOutputParser())

def split_query(state: RAGState):
    query = state["query"]
    sub_questions = decompose_chain.invoke({"query": query})
    # Convert to list
    sub_qs = [q.strip("-• \n") for q in sub_questions.split("\n") if q.strip()]
    return {**state, "sub_queries": sub_qs}


### ---------- 3. RAG for Each Sub-query ----------


embedding_fn = OllamaEmbeddings(model="nomic-embed-text")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rag_for_subquery(sub_query: str) -> Dict[str, Any]:
    docs = compression_retriever.base_retriever.get_relevant_documents(sub_query)  # skip compression for similarity

    query_embedding = embedding_fn.embed_query(sub_query)

    # Rerank docs based on similarity
    scored_docs = []
    for doc in docs:
        doc_embedding = embedding_fn.embed_query(doc.page_content)
        score = cosine_similarity(query_embedding, doc_embedding)
        scored_docs.append((score, doc))

    # Sort by similarity score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:3]]  # top 3

    # Final context
    context = "\n\n".join([doc.page_content for doc in top_docs])
    sources = [doc.metadata.get("source", "unknown") for doc in top_docs]

    answer_prompt = PromptTemplate.from_template(
        "Given the following context:\n\n{context}\n\nAnswer the question:\n{question}\nInclude sources in format [source]."
    )
    chain = (answer_prompt | llm | StrOutputParser())
    answer = chain.invoke({"context": context, "question": sub_query})

    return {"sub_query": sub_query, "answer": answer, "sources": list(set(sources))}

def rag_all(state: RAGState):
    sub_qs = state["sub_queries"]
    results = [rag_for_subquery(q) for q in sub_qs]
    return {**state, "answers": results}


### ---------- 4. Combine Final Answer ----------
def combine(state: RAGState):
    all_answers = state["answers"]
    final = ""
    seen_sources = set()

    for item in all_answers:
        final += f"Q: {item['sub_query']}\nA: {item['answer']}\n\n"
        seen_sources.update(item['sources'])

    final += "Sources:\n" + "\n".join(f"- {src}" for src in seen_sources)
    return {"final_answer": final}


### ---------- 5. LangGraph Assembly ----------
def build_graph():
    builder = StateGraph(RAGState)

    builder.add_node("split_query", RunnableLambda(split_query))
    builder.add_node("run_rag", RunnableLambda(rag_all))
    builder.add_node("combine", RunnableLambda(combine))

    builder.set_entry_point("split_query")
    builder.add_edge("split_query", "run_rag")
    builder.add_edge("run_rag", "combine")
    builder.add_edge("combine", END)

    return builder.compile()


graph = build_graph()


### ---------- 6. Pipeline Runner ----------
async def run_graph_pipeline(query: str) -> str:
    result = await graph.ainvoke({"query": query})
    return result["final_answer"]
