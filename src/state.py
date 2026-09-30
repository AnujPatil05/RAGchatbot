from typing import List, Dict, Any, TypedDict

class GraphState(TypedDict):
    question: str
    context: List[Dict[str, Any]]
    final_answer: str
    confidence_or_score: float