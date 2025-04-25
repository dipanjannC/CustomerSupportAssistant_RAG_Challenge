# TODO

# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph

# workflow = StateGraph(state_schema=MessagesState)


# # Define the function that calls the model
# def call_model(state: MessagesState):
#     system_prompt = (
#         "You are a helpful assistant. "
#         "Answer all questions to the best of your ability."
#     )
#     messages = [SystemMessage(content=system_prompt)] + state["messages"]
#     response = model.invoke(messages)
#     return {"messages": response}


# # Define the node and edge
# workflow.add_node("model", call_model)
# workflow.add_edge(START, "model")

# # Add simple in-memory checkpointer
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)