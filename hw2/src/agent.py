import os
import sys

# Ensure the root directory is in sys.path so 'hw2' module can be resolved
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import the tools we built
from hw2.src.tools import predict_galaxy_class, retrieve_domain_knowledge

# Ensure the user has their API key set
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY or GEMINI_API_KEY environment variable is missing!")
    api_key = "dummy_key_for_import"

# 1. Initialize the LLM (Gemini)
# We use gemini-2.5-flash as it is fast, free-tier eligible, and supports tools well.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=api_key)

# 2. Bind the tools to the LLM
tools = [retrieve_domain_knowledge, predict_galaxy_class]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the Agent Logic Node
def call_model(state: MessagesState):
    """Invokes the LLM to decide the next action or respond to the user."""
    # We prepend a system message to guide the agent's behavior
    system_prompt = (
        "You are a helpful astrophysics AI assistant specializing in galaxy formation and nuclear activity. "
        "You have access to two tools:\n"
        "1. retrieve_domain_knowledge: Use this to answer factual or conceptual questions about the domain.\n"
        "2. predict_galaxy_class: Use this to predict the nuclear activity class of a galaxy given numerical features.\n"
        "You MUST use these tools when appropriate. Do not guess information. "
        "Maintain a conversational tone and use previous context from the session memory if the user asks a follow-up question."
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

# 4. Define the routing logic
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Determines whether to call a tool or end the turn."""
    last_message = state["messages"][-1]
    # If the LLM decided to call a tool, route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, the LLM has finished responding
    return "__end__"

# 5. Build the StateGraph
workflow = StateGraph(MessagesState)

# Add the nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Add the edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "__end__": END
    }
)
workflow.add_edge("tools", "agent")

# 6. Setup Memory and Compile
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 7. Expose an easy entrypoint function for Task 4
def invoke_agent(message: str, session_id: str) -> str:
    """
    Invokes the agent with a message and maintains conversation memory using the session_id.
    """
    config = {"configurable": {"thread_id": session_id}}
    
    # We stream the events to just grab the final message, but we could also stream tokens
    final_state = app.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    
    return final_state["messages"][-1].content

if __name__ == "__main__":
    # Quick terminal testing
    import sys
    
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable first.")
        sys.exit(1)
        
    print("Testing Agent (type 'quit' to exit)...")
    session = "test_session_1"
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = invoke_agent(user_input, session)
        print(f"\nAgent: {response}")
