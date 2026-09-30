from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.graph import rag_app

app = FastAPI(
    title="Appening AI Engineer - Agentic AI RAG API",
    version="1.0.0",
    description="LangGraph-powered RAG pipeline answering strictly from the Agentic AI eBook."
)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    final_answer: str
    retrieved_context_chunks: List[Dict[str, Any]]
    confidence_or_score: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    initial_state = {
        "question": request.query,
        "context": [],
        "final_answer": "",
        "confidence_or_score": 0.0
    }
    
    result = rag_app.invoke(initial_state)
    
    return ChatResponse(
        final_answer=result["final_answer"],
        retrieved_context_chunks=result["context"],
        confidence_or_score=result["confidence_or_score"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)