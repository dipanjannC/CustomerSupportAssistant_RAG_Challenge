import streamlit as st
import asyncio
from src.backend.vectorstore import get_vectorstore_instance

# Set page configuration
st.set_page_config(
    page_title="Customer Support Assistant",
    page_icon="🔍",
    layout="wide"
)


# Initializing Vectorstore with caching
@st.cache_resource  
def initialize_vectorstore():
    return get_vectorstore_instance()

vectorstore = initialize_vectorstore()


async def main():
    
    query = st.text_input("Enter your query:")
    if st.button("Get Response"):
        results = vectorstore.query(query, top_k=5)
        st.write("Results:", results)

if __name__ == "__main__":
    asyncio.run(main())


