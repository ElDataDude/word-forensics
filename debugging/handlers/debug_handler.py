"""
Debug handler for Word Forensics analysis.
"""
import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from openai import OpenAI

class DebugHandler:
    """Context manager for handling debug operations."""
    
    def __init__(self):
        self.start_time = None
        self.debug_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.logs_dir = self.debug_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"debug_log_{self.timestamp}.json"
        self.analysis_file = self.logs_dir / f"debug_analysis_{self.timestamp}.txt"
        
        # Set up logging handler
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        # Remove existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers = []
        
        # Set up JSON file handler for debug logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "function": "%(funcName)s", "message": %(message)s}')
        )
        
        # Set up minimal console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Only show INFO and above in console
        console_handler.setFormatter(
            logging.Formatter('%(message)s')  # Minimal format for console
        )
        
        # Add handlers to root logger
        root_logger.setLevel(logging.DEBUG)  # Capture all levels
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Create separate logger for OpenAI to prevent API logging
        openai_logger = logging.getLogger("openai")
        openai_logger.handlers = []
        openai_logger.addHandler(file_handler)
        openai_logger.propagate = False  # Don't propagate to root logger
        
    def __enter__(self):
        """Enter the debug context."""
        self.start_time = time.time()
        logging.debug(json.dumps({
            "event": "debug_session_start",
            "context": {
                "timestamp": self.timestamp,
                "log_file": str(self.log_file),
                "analysis_file": str(self.analysis_file)
            }
        }))
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the debug context and generate analysis."""
        duration = time.time() - self.start_time
        
        # Log session end
        logging.debug(json.dumps({
            "event": "debug_session_end",
            "context": {
                "duration": duration,
                "success": exc_type is None
            }
        }))
        
        # Generate analysis report
        self.generate_analysis_report()
        
    def wrap(self, func: Callable) -> Any:
        """Wrap a function with debug logging."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logging.error(json.dumps({
                    "event": "error",
                    "error": str(e),
                    "traceback": str(exc_info=True)
                }))
                raise
        return wrapper
        
    def generate_analysis_report(self):
        """Generate analysis report using OpenAI."""
        try:
            # Read log file
            with open(self.log_file) as f:
                logs = [line.strip() for line in f if line.strip()]
            
            # Take only the last 100 log entries to stay within token limits
            logs = logs[-100:]
            log_text = "\n".join(logs)
            
            # Initialize OpenAI client
            client = OpenAI()
            
            # Generate analysis
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a log analysis expert. Analyze the following debug logs and provide insights about errors, warnings, and potential improvements. Focus on the most recent and important events."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these recent debug logs and provide a concise report:\n\n{log_text}"
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Save analysis
            analysis = response.choices[0].message.content
            with open(self.analysis_file, "w") as f:
                f.write(f"""
Debug Analysis Report - {self.timestamp}
==================================================

{analysis}

Note: This analysis is based on the most recent 100 log entries.
Full logs are available in: {self.log_file}
""")
            
            logging.info(f"Analysis saved to: {self.analysis_file}")
            
        except Exception as e:
            logging.error(f"Error generating analysis report: {str(e)}")
            # Create a basic report without OpenAI
            with open(self.analysis_file, "w") as f:
                f.write(f"""
Debug Analysis Report - {self.timestamp}
==================================================

Error generating AI analysis: {str(e)}

Please check the full debug logs at: {self.log_file}
""")


def debug_log(func: Callable) -> Callable:
    """Decorator for debug logging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging.debug(json.dumps({
                "event": "function_call",
                "function": func.__name__,
                "duration": duration,
                "success": True
            }))
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(json.dumps({
                "event": "function_call",
                "function": func.__name__,
                "duration": duration,
                "success": False,
                "error": str(e)
            }))
            raise
    return wrapper


def track_performance(func: Callable) -> Callable:
    """Decorator for performance tracking."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging.debug(json.dumps({
                "event": "performance",
                "function": func.__name__,
                "duration": duration
            }))
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(json.dumps({
                "event": "performance",
                "function": func.__name__,
                "duration": duration,
                "error": str(e)
            }))
            raise
    return wrapper
