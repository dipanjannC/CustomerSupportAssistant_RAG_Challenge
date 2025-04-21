import logging
import uuid
import coloredlogs

# 
correlation_id_storage = {}

def set_correlation_id(correlation_id: str=None):
    """Set the correlation ID for logging."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    else:
        correlation_id = str(correlation_id)
    correlation_id_storage['correlation_id'] = correlation_id
    return correlation_id


class CorrelationIDAdapter(logging.LoggerAdapter):
    """Logger adapter to add correlation ID to log records."""

    def process(self, msg, kwargs):
        """Add correlation ID to log records."""
        correlation_id = correlation_id_storage.get('correlation_id',None)

        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        msg = f"[{correlation_id}] {msg}"
        return msg, kwargs
    

def setup_logging():
    """Set up logging with colored output."""
    
    color_formatter = coloredlogs.ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level_styles={
            'debug': {'color': 'cyan'},
            'info': {'color': 'green'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red'},
            'critical': {'color': 'magenta', 'bold': True},
        }
    )

    #configure the root logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)

    #set the logging level (e.g. DEBUG for all levels)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # get root logger
    logger = logging.getLogger()

    # create logger adapter with correlationID
    adapter = CorrelationIDAdapter(logger, {})

    return adapter

    