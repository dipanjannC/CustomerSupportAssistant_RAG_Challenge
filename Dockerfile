# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /CustomerSupportAssistant_RAG_Challenge

# Copy the requirements file into the container
COPY ./requirements.txt /CustomerSupportAssistant_RAG_Challenge/requirements.txt

# Install any dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /CustomerSupportAssistant_RAG_Challenge/requirements.txt

# Copy the rest of the application code into the container
COPY ./src /CustomerSupportAssistant_RAG_Challenge/src

# Set PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH:/CustomerSupportAssistant_RAG_Challenge/src

# Expose the port FastAPI will run on
EXPOSE 8085

# Command to run the FastAPI app with Uvicorn
# CMD ["uvicorn", "src.app.api.app:app", "--host", "0.0.0.0", "--port", "8085","--reload"]

CMD uvicorn src.app.api.app:app --host 0.0.0.0 --port $PORT