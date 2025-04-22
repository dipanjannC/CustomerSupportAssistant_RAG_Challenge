# import os
# from phoenix.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor

# # 1. Export your Phoenix API key
# os.environ["PHOENIX_API_KEY"] = "YOUR_CLOUD_API_KEY"

# # 2. Register tracer (detects HTTP vs gRPC, path, and headers)
# tracer_provider = register(
#     project_name="customer-support-rag",
#     endpoint="https://app.phoenix.arize.com/v1/traces",  # full HTTP ingest path :contentReference[oaicite:3]{index=3}
#     protocol="http/protobuf",                             # enforce HTTP OTLP
#     verbose=True
# )

# # 3. Instrument your RAG/LLM calls
# LangChainInstrumentor().instrument(tracer_provider=tracer_provider)