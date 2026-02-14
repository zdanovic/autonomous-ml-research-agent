"""
AMRA: Autonomous ML Research Agent (v2.0.0)

A multi-agent system for autonomous Kaggle competition optimization.
"""
from .agents.data_scientist import DataScientist, DataScientistState
from .agents.evaluator import Evaluator, EvaluatorState
from .agents.experiment_designer import ExperimentDesigner, ExperimentDesignerState
from .agents.ml_engineer import MLEngineer, MLEngineerState
from .competition_config import CompetitionConfig, load_competition_data
from .core.observability import ObservabilityManager
from .core.orchestrator import Orchestrator

__all__ = [
    "CompetitionConfig",
    "DataScientist",
    "DataScientistState",
    "Evaluator",
    "EvaluatorState",
    "ExperimentDesigner",
    "ExperimentDesignerState",
    "MLEngineer",
    "MLEngineerState",
    "ObservabilityManager",
    "Orchestrator",
    "load_competition_data",
]
