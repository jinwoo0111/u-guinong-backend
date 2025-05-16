import os
from dotenv import load_dotenv
from fastapi import FastAPI

from models import Chat
from agent import get_graph

from langchain_core.messages import HumanMessage

load_dotenv()

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
LANGSMITH_TRACING=os.environ.get("LANGSMITH_TRACING").lower() == "true"
LANGSMITH_API_KEY=os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_ENDPOINT=os.environ.get("LANGSMITH_ENDPOINT")
LANGSMITH_PROJECT=os.environ.get("LANGSMITH_PROJECT")

graph = get_graph()
app = FastAPI()

@app.post("/chat")
async def chat(chat: Chat):
    input_messages = [HumanMessage(chat.content)]
    output = graph.invoke({
        "messages": input_messages,
        "input" :  chat.content
    })
    return output["messages"][-1]