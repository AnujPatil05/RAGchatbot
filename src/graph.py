from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes import retrieve_node, generate_node

def build_rag_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

rag_app = build_rag_graph()