import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🔍",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8085/generate_response"

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E90FF;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .response-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .history-item {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 5px;
        background-color: #e6f2ff;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">RAG Query System</p>', unsafe_allow_html=True)
st.markdown("Use this app to query the Retrieval-Augmented Generation (RAG) system.")

# Initialize session state for chat history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar for settings and features
with st.sidebar:
    st.markdown('<p class="sub-header">Settings</p>', unsafe_allow_html=True)
    
    # Option to clear history
    if st.button("Clear History", key="clear_history"):
        st.session_state.history = []
        st.success("History cleared!")
    
    # Display history statistics
    if st.session_state.history:
        st.markdown("### History Statistics")
        num_queries = len(st.session_state.history)
        avg_response_time = pd.Series([h.get('response_time', 0) for h in st.session_state.history]).mean()
        
        st.metric("Total Queries", num_queries)
        st.metric("Avg Response Time", f"{avg_response_time:.2f} sec")
    
    st.markdown("### About")
    st.info("""
    This application connects to a FastAPI backend that implements 
    a Retrieval-Augmented Generation (RAG) system for answering queries.
    """)

# Main query input area
st.markdown('<p class="sub-header">Ask a Question</p>', unsafe_allow_html=True)
query = st.text_area("Enter your query here:", height=100)

# Submit button for the query
if st.button("Submit Query", key="submit_query"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating response..."):
            try:
                # Record the start time
                start_time = datetime.now()
                
                # Send request to the API
                response = requests.post(
                    API_URL,
                    json={"query": query},
                    headers={"Content-Type": "application/json"}
                )
                
                # Record the end time and calculate response time
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                # Check response status
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add to history
                    history_entry = {
                        "query": query,
                        "response": result["response"],
                        "timestamp": result["timestamp"],
                        "response_time": response_time
                    }
                    st.session_state.history.append(history_entry)
                    
                    # Display the response
                    st.markdown('<div class="response-container">', unsafe_allow_html=True)
                    st.markdown("### Response:")
                    st.write(result["response"])
                    st.markdown(f"*Response time: {response_time:.2f} seconds*")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the API. Make sure the backend server is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Display history section
if st.session_state.history:
    st.markdown('<p class="sub-header">Query History</p>', unsafe_allow_html=True)
    
    # Create tabs for different history views
    tab1, tab2 = st.tabs(["List View", "Table View"])
    
    with tab1:
        # Reverse to show newest first
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Query {len(st.session_state.history) - i}: {item['query'][:50]}..."):
                st.write(f"**Query:**\n{item['query']}")
                st.write(f"**Response:**\n{item['response']}")
                st.write(f"**Timestamp:** {item['timestamp']}")
                st.write(f"**Response Time:** {item['response_time']:.2f} seconds")
    
    with tab2:
        # Create a dataframe for the table view
        history_data = []
        for i, item in enumerate(st.session_state.history):
            history_data.append({
                "Index": i + 1,
                "Query": item['query'][:50] + "..." if len(item['query']) > 50 else item['query'],
                "Timestamp": item['timestamp'],
                "Response Time (s)": f"{item['response_time']:.2f}"
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*RAG Query System - Powered by FastAPI and Streamlit*")