from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor,BatchSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor

import os

tracer_provider = trace_sdk.TracerProvider()
span_exporter = OTLPSpanExporter(
    endpoint=f"{os.getenv("PHOENIX_COLLECTOR_ENDPOINT")}/v1/traces",
    headers={
    "authorization": f"Bearer {os.getenv('PHOENIX_CLIENT_HEADERS')}"}
)

# Attach BatchSpanProcessor
processor = BatchSpanProcessor(span_exporter)
tracer_provider.add_span_processor(processor)
trace.set_tracer_provider(tracer_provider)

# Get a tracer instance
tracer = trace.get_tracer(__name__)
# Instrument LangChain
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

