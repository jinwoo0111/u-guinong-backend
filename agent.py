from state import State
from nodes import call_model
from langgraph.graph import StateGraph, START, END

from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

import json
from src.inference.embedding_model import embedding_model
from src.search.section_coarse_search import coarse_search_sections
from src.search.fine_search import fine_search_chunks

SECTIONS_PATH = "data/extracted/sections_with_emb.json"
CHUNK_INDEX_PATH = "data/index/full_vectors.json"
TOP_K_SECTIONS = 3
TOP_K_CHUNKS = 5

class Route(BaseModel):
    step: Literal["RAG", "LLM"] = Field(None)

llm = init_chat_model("gpt-4o-mini")
router = llm.with_structured_output(Route)

def call_router(state: State):
    decision = router.invoke([
            SystemMessage(
                content="Route the input to RAG if it is related to crops. If not, route the input to LLM."
            ),
            HumanMessage(content=state["input"]),
        ])
    return {"decision": decision.step}

def call_rag(state: State):
    return {"messages": ["rag has called"]}

def call_llm(state: State):
    pass

def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "RAG":
        return "RAG"
    elif state["decision"] == "LLM":
        return "LLM"


def get_graph():
    workflow = StateGraph(State)

    workflow.add_node("router", call_router)
    workflow.add_node("rag", call_rag)
    workflow.add_node("llm", call_model)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "RAG": "rag",
            "LLM": "llm",
        }
    )
    workflow.add_edge("rag", "llm")
    workflow.add_edge("llm", END)

    graph = workflow.compile()
    return graph

def get_document(State: State):
    query = State["messages"][-1]
    query_emb = embedding_model.get_embedding(query)
    with open(SECTIONS_PATH, "r", encoding="utf-8") as f:
        section_data = json.load(f)

    top_sections = coarse_search_sections(query, section_data, top_k=TOP_K_SECTIONS)

    
    with open(CHUNK_INDEX_PATH, "r", encoding="utf-8") as f:
        chunk_index = json.load(f)

    
    top_chunks = fine_search_chunks(query_emb, chunk_index, target_sections=top_sections, top_k=TOP_K_CHUNKS)

    prompt = "\n\n".join([
        f"[출처: {c['metadata']['source_pdf']}, section: {c['metadata']['section']}] {c['metadata']['text']}"
        for c in top_chunks
    ])


