"""
Structured Logger for the Agentic System.
"""
import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if passed
        if hasattr(record, "extra"):
            log_record.update(record.extra)
            
        return json.dumps(log_record)

def setup_logger(
    name: str = "hackathon_agent",
    log_file: str | None = "agent_execution.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a structured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console Handler (Human readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File Handler (JSON structured)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
        
    return logger

# Global logger instance
logger = setup_logger()
