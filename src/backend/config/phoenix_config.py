
from phoenix.otel import register 
import os

from dotenv import load_dotenv
load_dotenv()

tracer_provider = register(
    project_name="customer_support_assistant",  # Specify your project name here
    endpoint=f"{os.getenv('PHOENIX_DOCKER_COLLECTOR_ENDPOINT')}/v1/traces",
    batch=True,
    set_global_tracer_provider=True,
    auto_instrument=True,
)

# Get a tracer instance
tracer = tracer_provider.get_tracer(__name__)