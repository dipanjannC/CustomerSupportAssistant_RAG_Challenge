
from pathlib import Path
from functools import wraps
import time
from typing import Any
import json
from rich import print



project_root = Path(__file__).resolve().parents[3]


def execution_time(func):
    @wraps(func)  # Preserves the original function's metadata
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Capture the end time
        elapsed_time = end_time - start_time  # Calculate the time difference
        print(
            f"[bold blue]Execution time of {func.__name__}: {elapsed_time:.4f} seconds[/bold blue]"
        )
        return result  # Return the result of the original function

    return wrapper

def json_serializable(item: Any) -> Any:
    """Convert any non-JSON serializable objects to strings."""
    try:
        # Test if the item is JSON serializable
        json.dumps(item)
        return item
    except (TypeError, OverflowError):
        # If not serializable, convert to string
        return str(item)