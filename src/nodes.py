import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from src.config import PINECONE_API_KEY
from src.state import GraphState

INDEX_NAME = "agentic-ai-384"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)
def retrieve_node(state: GraphState):
    query = state["question"]
    query_vector = embeddings.embed_query(query)
    
    search_res = index.query(
        vector=query_vector,
        top_k=4,
        include_metadata=True
    )
    
    retrieved_chunks = []
    scores = []
    for match in search_res["matches"]:
        retrieved_chunks.append({
            "chunk_id": match["id"],
            "score": round(float(match["score"]), 4),
            "page": match["metadata"].get("page", "Unknown"),
            "content": match["metadata"].get("text", "")
        })
        scores.append(float(match["score"]))
        
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {"context": retrieved_chunks, "confidence_or_score": avg_score}

def generate_node(state: GraphState):
    question = state["question"]
    context_chunks = state.get("context", [])
    
    system_prompt = (
        "You are an expert technical AI assistant answering questions strictly based on the provided "
        "Agentic AI eBook context. Do not speculate or use outside knowledge. If the answer is not present "
        "in the context, state clearly: 'I cannot answer this based on the provided Agentic AI eBook knowledge base.'"
    )
    
    formatted_context = "\n\n".join(
        [f"[Page {c['page']}] {c['content']}" for c in context_chunks]
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": formatted_context, "question": question})
    
    return {"final_answer": response.content}