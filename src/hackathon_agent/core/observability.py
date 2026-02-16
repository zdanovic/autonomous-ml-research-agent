"""
Observability Manager.

Wraps Comet ML and Opik for unified experiment tracking and tracing.
"""
import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import contextlib
import uuid

# Graceful imports in case packages are not installed yet
try:
    import comet_ml
    from comet_ml import Experiment, OfflineExperiment
except ImportError:
    comet_ml = None
    Experiment = None
    OfflineExperiment = None

try:
    import opik
    from opik import Opik, track
except ImportError:
    opik = None
    track = None

logger = logging.getLogger(__name__)

class ObservabilityManager:
    """
    Manages Comet ML experiments and Opik traces.
    """
    
    def __init__(self, project_name: str = "amra-research", workspace: str | None = None):
        self.project_name = project_name
        self.workspace = workspace
        self.comet_experiment = None
        self.opik_client = None
        
        # Initialize Comet
        self._init_comet()
        
        # Initialize Opik
        self._init_opik()
        
    def _init_comet(self):
        """Initialize Comet ML Experiment."""
        api_key = os.getenv("COMET_API_KEY")
        if not api_key:
            logger.warning("COMET_API_KEY not found. Observability will be limited.")
            return

        if comet_ml:
            try:
                # We use OfflineExperiment if internet is flaky, but Online is better for agents
                self.comet_experiment = Experiment(
                    api_key=api_key,
                    project_name=self.project_name,
                    workspace=self.workspace,
                    log_code=True,
                    auto_metric_logging=True
                )
                logger.info("Comet ML initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Comet: {e}")
                
    def _init_opik(self):
        """Initialize Opik for Tracing."""
        # Opik usually auto-configures from OPIK_API_KEY env var
        if opik and os.getenv("OPIK_API_KEY"):
            try:
                self.opik_client = Opik(project_name=self.project_name)
                logger.info("Opik initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Opik: {e}")
        else:
             logger.warning("Opik not configured (missing OPIK_API_KEY or package).")

    def log_metric(self, name: str, value: float, step: int | None = None):
        """Log a metric to Comet."""
        if self.comet_experiment:
            self.comet_experiment.log_metric(name, value, step=step)
            
    def log_parameters(self, params: dict[str, Any]):
        """Log parameters to Comet."""
        if self.comet_experiment:
            self.comet_experiment.log_parameters(params)
            
    @contextmanager
    def trace(self, name: str, input: Any = None):
        """
        Context manager for Opik tracing. 
        If Opik is not available, it's a no-op pass-through.
        """
        if opik and track:
            # We can use the decorator or manual span creation.
            # ideally dspy integrations handles this, but for manual agent steps:
            # For now, simplistic pass-through or pseudo-trace
            # Real Opik usage often involves decorating functions
            yield
        else:
            yield

    def log_feedback(self, score: float, name: str = "performance", trace_id: str | None = None):
        """
        Log feedback score for a trace (or current context).
        """
        if opik and self.opik_client:
            # If trace_id is not provided, Opik might attach to current context if called within a trace
            # For now, we assume simple logging.
            # In a real impl, we'd need to store the trace_id from the context manager.
            # providing a simplified placeholder
            with contextlib.suppress(Exception):
                # self.opik_client.log_feedback(...) depending on SDK version
                pass

    def end(self):
        """End experiment."""
        if self.comet_experiment:
            self.comet_experiment.end()
