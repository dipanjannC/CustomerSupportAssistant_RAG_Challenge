# Customer Assistant RAG Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for building a customer support chatbot.

---

## Features

- **Preprocessing and Understanding**:
  - Includes data understanding, parsing and processing strategies for feature
  creation and initializing vector databases.
- **Retrieval-Augmented Generation (RAG)**:
  - Combines document retrieval with LLM-based response generation.
- **Customizable Prompts**:
  - Uses system, retriever, and user prompts for better control over the chatbot's behavior.
- **Evaluation Tools**:
  - Includes scripts for dataset validation, evaluation metrics, and pipeline testing.
- **Streamlit UI**:
  - A user-friendly interface for interacting with the chatbot.
- **FastAPI Backend**:
  - Provides an API for generating responses programmatically.

---

## Prerequisites

1. **Python Version**:
   - Ensure you have Python 3.12 installed.
2. **Install Dependencies**:
   - Use `pip` to install the required dependencies from `requirements.txt`.
3. **Set Python Path**:
   - Setting the Python path is crucial for importing packages correctly.

---

## Setting Up the Application

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/CustomerSupportAssistant_RAG_Challenge.git
cd CustomerSupportAssistant_RAG_Challenge
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate 
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r [requirements.txt]
```

### 4. Set the Python Path

To ensure that all modules are imported correctly, set the PYTHONPATH environment variable:

For mac and linux

```bash
export PYTHONPATH=$(pwd)
```

For Windows:

```bash
set PYTHONPATH=%cd%
```

## Running the Application

### 1. Start the Backend (FastAPI)

Run the FastAPI backend to expose the chatbot API:

```bash
# python src/backend/api/app.py
uvicorn src.backend.api.app:app --host 0.0.0.0 --port 8085
```

The API will be available at [http://localhost:8085](http://localhost:8085).

### 2. Start the Streamlit UI

Run the Streamlit application for a user-friendly interface:

```bash
streamlit run src/ui/streamlit_app.py 
```

The UI will be available at [http://localhost:8501](http://localhost:8501).

## Important Notes

- **Setting Python Path**:

 Always set the `PYTHONPATH` environment variable before running the application to avoid import errors.

- **Docker Support**:

A Dockerfile is included for containerizing the application. Future work includes adding docker-compose for connecting the UI and backend.

## Future Work

- Add Docker Compose for seamless integration of UI and backend.
- Implement evaluation metrics for generation and retrieval using RAGAS.
- Enhance the chatbot with Chain-of-Thought reasoning for better explanations.
- Expand support for multilingual queries.
- Add test cases for each functionalities.

## Limitations

- Currently supports only English queries.
- Requires a pre-configured vectorstore for document retrieval.

## Additional Resources

some interesting repositories

- [Awesome LLM Interpretability](https://github.com/JShollaj/awesome-llm-interpretability)
- [Prompting Guide](https://www.promptingguide.ai/research/llm-reasoning)
- [System Prompts and Models (hidden system prompts from cursor and more systems)](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools)
- [Camel AI](https://github.com/camel-ai/camel/)
