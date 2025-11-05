from typing import TypedDict, Literal
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import os

class Emailtonerstate(TypedDict):
    tonetype: bool  # True = formal, False = professional
    message: str
    final_message: str

load_dotenv()

model_name = os.getenv("MODEL_NAME")
model_provider = os.getenv("MODEL_PROVIDER")
llm = init_chat_model(model=model_name, model_provider=model_provider)

def formal_message(state: Emailtonerstate):
    prompt = f"You are an expert in writing formal emails. Convert the below message into a formal tone:\n\n{state['message']}"
    state["final_message"] = llm.invoke(prompt)
    return state

def professional_message(state: Emailtonerstate):
    prompt = f"You are an expert in writing professional emails. Convert the below message into a professional tone:\n\n{state['message']}"
    state["final_message"] = llm.invoke(prompt)
    return state

def route_message(state: Emailtonerstate) -> Literal["formal", "professional"]:
    if state["tonetype"]:
        return "formal"
    else:
        return "professional"

email_tone_graph = StateGraph(Emailtonerstate)
email_tone_graph.add_node("formal", formal_message)
email_tone_graph.add_node("professional", professional_message)
email_tone_graph.add_conditional_edges(START, route_message)
email_tone_graph.add_edge("formal", END)
email_tone_graph.add_edge("professional", END)

graph = email_tone_graph.compile()

