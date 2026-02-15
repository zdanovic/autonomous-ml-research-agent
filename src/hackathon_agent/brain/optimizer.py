"""
DSPy Optimizer Module.

Responsibility: "Compile" the agents by learning from successful experiments.
"""
import logging

import dspy
from dspy.teleprompt import BootstrapFewShot

from ..schema.experiment import ExperimentResult, Hypothesis
from .hypothesis_generation import (
    DataSummary,
    GeneratedHypothesis,
    HypothesisGenerator,
    HypothesisList,
)

logger = logging.getLogger(__name__)

class AgentOptimizer:
    """
    Optimizes DSPy modules using historical experiment data.
    """
    
    def __init__(self):
        self.teacher = HypothesisGenerator()
        
    def create_examples(self, 
                       data_summary: DataSummary, 
                       history: list[ExperimentResult], 
                       hypotheses_map: dict[str, Hypothesis]) -> list[dspy.Example]:
        """
        Convert experiment history into DSPy training examples.
        Only keeps successful experiments.
        """
        successful_hypotheses = []
        
        # Filter for success (e.g., score better than baseline or simply completed successfully)
        # In a real scenario, we'd compare against a baseline score.
        # Here we take top 50% of results or just valid ones.
        if not history:
            return []
            
        # sorted_history = sorted(history, key=lambda x: x.cv_score)
        # top_results = sorted_history[:len(history)//2] # Top 50%
        
        for res in history:
            # Simple heuristic: if it didn't fail and score is decent
            if res.success and res.hypothesis_id in hypotheses_map:
                orig_hyp = hypotheses_map[res.hypothesis_id]
                successful_hypotheses.append(
                    GeneratedHypothesis(
                        id=orig_hyp.id,
                        statement=orig_hyp.statement,
                        rationale=orig_hyp.rationale,
                        experiment_design=orig_hyp.experiment_design
                    )
                )
                
        if not successful_hypotheses:
            return []

        # Create one example where Input=DataSummary, Output=List[GoodHypotheses]
        # This teaches the model: "Given this data, YOU SHOULD HAVE Outputted THESE hypotheses"
        example = dspy.Example(
            data_summary=data_summary,
            hypotheses=HypothesisList(hypotheses=successful_hypotheses)
        ).with_inputs("data_summary")
        
        return [example]

    def compile(self, train_examples: list[dspy.Example], save_path: str = "src/hackathon_agent/brain/compiled_hypothesizer.json"):
        """
        Compile the HypothesisGenerator using BootstrapFewShot.
        """
        if not train_examples:
            logger.warning("No training examples for optimization.")
            return self.teacher

        logger.info(f"Compiling agent with {len(train_examples)} examples...")
        
        # Simple metric: checks if the output structure is valid
        # (Optimization relies more on the Few-Shot demonstrations than the metric here)
        def validate_structure(example, pred, trace=None):
            return isinstance(pred.hypotheses, HypothesisList) and len(pred.hypotheses.hypotheses) > 0

        # Create Teleprompter
        # max_bootstrapped_demos=2 means it will add up to 2 full Input/Output pairs to the prompt
        teleprompter = BootstrapFewShot(metric=validate_structure, max_bootstrapped_demos=2)
        
        try:
            compiled_program = teleprompter.compile(self.teacher, trainset=train_examples)
            compiled_program.save(save_path)
            logger.info(f"Optimized agent saved to {save_path}")
            return compiled_program
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self.teacher
