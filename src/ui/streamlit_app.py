import streamlit as st

def main():
    st.title("Simple QA App")
    question = st.text_input("Enter your question:")

    if question:
        # Replace this with actual QA logic
        qa_pairs = {
            "What is Streamlit?": "Streamlit is an open-source Python library that makes it easy to create interactive web apps for data science and machine learning.",
            "How do I run a Streamlit app?": "Run your app using `streamlit run your_app.py`.",
            "What is this app?": "This is a very basic QA app built with Streamlit for demonstration purposes."
        }
        if question in qa_pairs:
            answer = qa_pairs[question]
            st.write(f"Answer: {answer}")
        else:
            st.write("I don't know the answer to that question.")

if __name__ == "__main__":
    main()


# import streamlit
# import src.ui.streamlit_app

# if __name__ == "__main__":
#     src.ui.streamlit_app.main()
