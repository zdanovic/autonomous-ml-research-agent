"""
DSPy Module for Experiment Design.
"""
from typing import Any

import dspy
from pydantic import BaseModel, Field


class ExperimentSpecOutput(BaseModel):
    name: str = Field(..., description="Short name for the experiment")
    description: str = Field(..., description="Detailed description")
    strategy: str = Field(..., description="One of: catboost_grouped, lightgbm_grouped, target_encoding, log_target")
    params: dict[str, Any] = Field(..., description="Parameters for the strategy")
    validation_strategy: str = Field("kfold", description="Validation strategy")

class DesignExperimentSignature(dspy.Signature):
    """
    Given a scientific hypothesis, design a concrete machine learning experiment to test it.
    Choose the appropriate strategy and parameters.
    """
    hypothesis_statement: str = dspy.InputField(desc="The hypothesis to test")
    hypothesis_rationale: str = dspy.InputField(desc="Why this hypothesis is proposed")
    experiment_spec: ExperimentSpecOutput = dspy.OutputField(desc="Executable experiment specification")

class ExperimentDesignerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(DesignExperimentSignature)
        
    def forward(self, statement: str, rationale: str) -> ExperimentSpecOutput:
        return self.prog(hypothesis_statement=statement, hypothesis_rationale=rationale).experiment_spec
