from cache import cache_assessment, get_cached_assessment
from retriever import hybrid_retrieve
from reranker import rerank
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from tools import fetch_from_wikipedia
from cache import get_user_difficulty

groq = ChatGroq(temperature=0.3, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

PROMPT = PromptTemplate.from_template("""
You are an expert assessment generator. Based on the context below and the topic: "{topic}", learning objectives: {objectives}, and difficulty: {difficulty},
create a personalized quiz with 5 diverse questions:
- Include MCQs, True/False, Short Answer
- Provide correct answers and brief explanations

Context:
{context}
""")

async def generate_assessment(request):
    # Step 1: Check Cache
    cache_hit = get_cached_assessment(request.model_dump())
    if cache_hit:
        return {"cached": True, "assessment": cache_hit["assessment"]}

    # Step 2: Retrieve + Rerank

# Step 2: Retrieve + Rerank
    context_docs = hybrid_retrieve(request.topic)

    if not context_docs:
        context = fetch_from_wikipedia(request.topic)
    else:
        reranked_docs = rerank(request.topic, context_docs)
    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    difficulty = request.difficulty
    if difficulty == "auto":
        difficulty = get_user_difficulty(request.user_id)

    print(context,"LLMCONTEXT")
    # Step 3: Generate
    chain = PROMPT | groq
    result = chain.invoke({
        "topic": request.topic,
        "objectives": request.objectives,
        "difficulty": difficulty,
        "context": context,
    })

    response = {"assessment": result.content}
    cache_assessment(request.model_dump(), response)
    return response
