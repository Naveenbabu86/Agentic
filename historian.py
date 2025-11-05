from typing import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
import os
from datetime import date

class History(TypedDict):
    draft: date
    
    history: str

load_dotenv()
model_name = os.getenv('MODEL_NAME')
llm = init_chat_model(model=model_name)

def historical_events(state: History) -> History:
    draft = state['draft']
    prompt = ChatPromptTemplate([
        ('system', 'You are an expert remembering all the historical events happend in india'),
        ("user", "When you give the date {draft}, get the events that happened on that particular day."),
        
    ])
    chain = prompt | llm 
    response = chain.invoke({'draft': draft})
    state['history'] = response.content
    return state

# ✅ Build the state graph properly
historian_graph = StateGraph(History)
historian_graph.add_node("historian", historical_events)  # ✅ fixed
historian_graph.add_edge(START, "historian")
historian_graph.add_edge("historian", END)

graph = historian_graph.compile()
