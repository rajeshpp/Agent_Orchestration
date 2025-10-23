from langgraph.graph import StateGraph
from typing import TypedDict
from .llm_nodes import preprocess_node, analysis_node, explanation_node


class BrainTumorState(TypedDict):
    """Schema of data passed between graph nodes."""
    patient_data: dict
    normalized_text: str
    risk_summary: str
    final_report: str


def build_graph():
    # Define the data structure flowing between nodes
    graph = StateGraph(BrainTumorState)

    # Add processing nodes
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("analyze", analysis_node)
    graph.add_node("explain", explanation_node)

    # Define the data flow edges
    graph.add_edge("preprocess", "analyze")
    graph.add_edge("analyze", "explain")

    # Define entry and finish points
    graph.set_entry_point("preprocess")
    graph.set_finish_point("explain")

    return graph
