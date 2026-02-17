"""
Orchestrator Agent.

Responsibility: Manage the lifecycle of the autonomous research process.
"""
from typing import Any

from ..agents.data_scientist import DataScientist
from ..agents.evaluator import Evaluator
from ..agents.experiment_designer import ExperimentDesigner
from ..agents.ml_engineer import MLEngineer
from ..schema.experiment import ExperimentResult
from .agent import Agent, AgentState
from .observability import ObservabilityManager


class OrchestratorState(AgentState):
    iteration: int = 0
    best_score: float = float("inf")
    history: list[ExperimentResult] = []


class Orchestrator(Agent[OrchestratorState]):
    """
    Main controller for the autonomous research loop.
    """
    
    def __init__(self, name: str = "Orchestrator"):
        state = OrchestratorState(name=name)
        super().__init__(name, state)
        
        # Initialize Observability (Comet + Opik)
        self.observability = ObservabilityManager(project_name="amra-research")
        
        # Initialize sub-agents with their specific states
        from ..agents.data_scientist import DataScientistState
        from ..agents.evaluator import EvaluatorState
        from ..agents.experiment_designer import ExperimentDesignerState
        from ..agents.ml_engineer import MLEngineerState
        
        self.data_scientist = DataScientist(
            "DataScientist", 
            state=DataScientistState(name="DataScientist")
        )
        self.designer = ExperimentDesigner(
            "ExperimentDesigner", 
            state=ExperimentDesignerState(name="ExperimentDesigner")
        )
        self.ml_engineer = MLEngineer(
            "MLEngineer", 
            state=MLEngineerState(name="MLEngineer")
        )
        self.evaluator = Evaluator(
            "Evaluator",
            state=EvaluatorState(name="Evaluator")
        )
        
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the main research loop.
        """
        max_iterations = context.get("max_iterations", 5)
        
        self.log(f"Starting autonomous research loop. Max iterations: {max_iterations}")
        self.observability.log_parameters({"max_iterations": max_iterations})
        
        try:
            for i in range(max_iterations):
                self.state.iteration = i + 1
                self.log(f"--- Iteration {self.state.iteration}/{max_iterations} ---")
                
                # 1. Data Analysis & Hypothesis Generation
                self.log("Step 1: Data Scientist Analysis")
                with self.observability.trace("Data Scientist Analysis", input=context):
                    ds_result = self.data_scientist.run(context)
                context["hypotheses"] = ds_result["hypotheses"]
                self.observability.log_metric("hypotheses_count", len(ds_result["hypotheses"]), step=i)
                
                # 2. Experiment Design
                self.log("Step 2: Experiment Design")
                with self.observability.trace("Experiment Design", input=context):
                    des_result = self.designer.run(context)
                context["planned_experiments"] = des_result["new_experiments"]
                self.observability.log_metric("planned_experiments_count", len(des_result["new_experiments"]), step=i)
                
                # 3. Execution
                if not context["planned_experiments"]:
                    self.log("No new experiments planned. Stopping early.")
                    break
                    
                self.log("Step 3: ML Engineering Execution")
                with self.observability.trace("ML Execution", input=context):
                    ml_result = self.ml_engineer.run(context)
                new_results = ml_result["results"]
                
                # 4. Learning & Update
                self.state.history.extend(new_results)
                context["history"] = self.state.history
                
                # 5. Evaluation & Leaderboard
                self.log("Step 4: Evaluation & Leaderboard")
                with self.observability.trace("Evaluation", input=context):
                    eval_result = self.evaluator.run(context)
                
                best_score = eval_result["best_score"]
                self.observability.log_metric("best_cv_score", best_score, step=i)
                self._update_best_score(new_results, best_score)
                
                # Clear planned experiments for next loop
                context["planned_experiments"] = []
                
                # 6. Self-Optimization (The "Learning" Step)
                if i > 0 and i % 2 == 0: # Optimize every 2 iterations
                    self.log("Step 5: Self-Optimization")
                    self._optimize_agents()
                
        finally:
            self.observability.end()
            
        return {
            "best_score": self.state.best_score,
            "history": self.state.history
        }

    def _optimize_agents(self):
        """
        Run DSPy optimizer to improve agent prompts based on successful experiments.
        """
        try:
            from ..brain.config import init_dspy
            from ..brain.optimizer import AgentOptimizer
            
            # Ensure stats are available
            if not self.data_scientist.state.last_data_summary:
                self.log("Skipping optimization: No Data Summary available.", level="WARNING")
                return
                
            init_dspy()
            optimizer = AgentOptimizer()
            
            # Create training examples from history
            # We need a map of ID -> Hypothesis to reconstruct the full hypothesis object from history
            # This is a simplification; ideally history stores full objects or we fetch from DS state
            all_hypotheses = {h.id: h for h in self.data_scientist.state.hypotheses}
            
            examples = optimizer.create_examples(
                data_summary=self.data_scientist.state.last_data_summary,
                history=self.state.history,
                hypotheses_map=all_hypotheses
            )
            
            if examples:
                self.log(f"Compiling Agent with {len(examples)} successful examples...")
                optimizer.compile(examples)
            else:
                self.log("No successful examples found for optimization yet.")
                
        except Exception as e:
            self.log(f"Optimization failed: {e}", level="ERROR")
            
    def _update_best_score(self, results: list[ExperimentResult], current_best: float):
        # Allow best score to be updated either from local batch or global evaluator
        if current_best < self.state.best_score:
            self.state.best_score = current_best
            self.log(f"🎉 NEW BEST SCORE confirmed by Evaluator: {self.state.best_score:.4f}")
            
            # Log positive feedback to Opik for this successful iteration
            self.observability.log_feedback(score=1.0, name="new_best_score")
