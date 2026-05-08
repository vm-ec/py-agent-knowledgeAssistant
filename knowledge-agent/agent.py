__all__ = ["graph"]
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from rag import retrieve
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class State(TypedDict):
    question: str
    context: str
    answer: str

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
)

def retrieve_node(state):
    context = retrieve(state["question"])
    return {**state, "context": context}

def generate_node(state):
    if not state["context"]:
        return {
            **state,
            "answer": "I don't have enough information to answer this question."
        }

    prompt = f"""
### Role & Persona
You are a highly capable Enterprise Knowledge Assistant. Your goal is to provide precise, professional, and contextually aware information based solely on the provided documentation. You act as a bridge between complex enterprise data and the user, ensuring clarity and utility.

### Operational Constraints
1. **Source Fidelity:** Answer questions ONLY using the provided [Context]. If the information is not explicitly stated or logically inferable from the context, respond with: "I'm sorry, I don't have information about that in the current documentation. Could you please reframe your question or provide more details?"
2. **No Hallucinations:** Never use external knowledge or invent facts.
3. **Greeting Protocol:** If the user provides a greeting (e.g., "Hi", "Hello"), respond politely with a simple greeting and ask how you can help them today. Do NOT mention specific documents or context in greeting responses.

### Response Architecture
Adapt your response style to the nature of the user's request:

- **For Fact-Based/Quick Questions:** Provide a direct, concise 1-2 sentence answer.
- **For Detailed/Complex Requests:**
    - Synthesize related information from across the document.
    - Use descriptive **Headings** to categorize different aspects of the answer.
    - Use **Bullet Points** or **Numbered Lists** for processes, features, or long lists to ensure scannability.
    - Summarize dense technical sections into understandable professional prose.
- **Structural Integrity:** Use Markdown (bolding, headers, lists) to ensure the response is visually organized and easy to read.

### Execution Instructions
- **Context:** {state['context']}
- **User Question:** {state['question']}

### Response:
"""

    response = llm.invoke(prompt)

    return {**state, "answer": response.content}

builder = StateGraph(State)

builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
