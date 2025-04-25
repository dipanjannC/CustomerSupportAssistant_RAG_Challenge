FROM python:3.12-alpine

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# If you need a proxy, uncomment these lines:
# ARG HTTP_PROXY
# ARG HTTPS_PROXY
# ENV http_proxy=${HTTP_PROXY}
# ENV https_proxy=${HTTPS_PROXY}

WORKDIR /CustomerSupportAssistant_RAG_Challenge

# Switch HTTP → HTTPS in apt sources (optional but helps avoid interception)
RUN sed -i \
      -e 's|http://deb.debian.org/|https://deb.debian.org/|g' \
      /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        build-essential \
        libpq-dev \
        libffi-dev \
        libssl-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY src src

ENV PYTHONPATH=/CustomerSupportAssistant_RAG_Challenge/src

EXPOSE 8085

# add -- reload to enable hot reloading
CMD ["uvicorn", "src.backend.api.app:app", "--host", "0.0.0.0", "--port", "8085"]
