"""
Experiment Designer Agent.

Responsibility: Translate abstract hypotheses into concrete execution plans.
"""
import uuid
from typing import Any

from ..core.agent import Agent, AgentState
from ..schema.experiment import ExperimentSpec, ExperimentStatus, Hypothesis


class ExperimentDesignerState(AgentState):
    planned_experiments: list[ExperimentSpec] = []

class ExperimentDesigner(Agent[ExperimentDesignerState]):
    """
    Agent that designs experiments based on hypotheses.
    """
    
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Design experiments for untested hypotheses.
        """
        hypotheses: list[Hypothesis] = context.get("hypotheses", [])
        
        new_experiments = []
        
        for hypothesis in hypotheses:
            if hypothesis.status == "untested":
                experiment = self._design_experiment(hypothesis)
                if experiment:
                    new_experiments.append(experiment)
                    hypothesis.status = "planned"
        
        self.state.planned_experiments.extend(new_experiments)
        self.log(f"Designed {len(new_experiments)} new experiments.")
        
        return {
            "new_experiments": new_experiments
        }

    def _design_experiment(self, hypothesis: Hypothesis) -> ExperimentSpec | None:
        """
        Map hypothesis to a specific experiment configuration.
        """
        self.log(f"Designing experiment for: {hypothesis.id}")
        
        # Try DSPy
        try:
            from ..brain.config import init_dspy
            from ..brain.experiment_design import ExperimentDesignerModule
            
            init_dspy()
            
            designer = ExperimentDesignerModule()
            spec_out = designer(statement=hypothesis.statement, rationale=hypothesis.rationale)
            
            exp = ExperimentSpec(
                id=f"EXP_{uuid.uuid4().hex[:8]}",
                hypothesis_id=hypothesis.id,
                name=spec_out.name,
                description=spec_out.description,
                strategy=spec_out.strategy,
                params=spec_out.params,
                validation_strategy=spec_out.validation_strategy,
                status=ExperimentStatus.PLANNED
            )
            return exp
            
        except Exception as e:
            self.log(f"DSPy design failed: {e}. Falling back to heuristics.", level="WARNING")

        # Fallback Heuristics
        strategy = "generic"
        params = {}

        # Heuristic mapping (Replace with smarter logic later)
        if "target encoding" in hypothesis.statement.lower():
            strategy = "feature_engineering"
            params = {"method": "target_encoding", "smoothing": 10}
        
        elif "log-transform" in hypothesis.statement.lower():
            strategy = "target_transform"
            params = {"transform": "log1p"}
            
        elif "country" in hypothesis.statement.lower():
            strategy = "catboost_grouped"
            params = {"group_col": "country"}
            
        exp = ExperimentSpec(
            id=f"EXP_{uuid.uuid4().hex[:8]}",
            hypothesis_id=hypothesis.id,
            name=f"Test {hypothesis.id}",
            description=hypothesis.experiment_design,
            strategy=strategy,
            params=params,
            status=ExperimentStatus.PLANNED
        )
        
        return exp
