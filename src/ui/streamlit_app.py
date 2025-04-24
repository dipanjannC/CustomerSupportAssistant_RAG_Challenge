import streamlit as st
import asyncio
from src.backend.vectorstore import get_vectorstore_instance

# Initialize Vectorstore
# Initialize Vectorstore with caching
@st.cache_resource  
def initialize_vectorstore():
    return get_vectorstore_instance()

vectorstore = initialize_vectorstore()


async def main():
    st.title("Customer Support Assistant")
    

    query = st.text_input("Enter your query:")
    if st.button("Get Response"):
        results = await vectorstore.query(query, top_k=5)
        st.write("Results:", results)

if __name__ == "__main__":
    asyncio.run(main())


