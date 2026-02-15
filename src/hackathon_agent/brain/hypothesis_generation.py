"""
DSPy Module for Hypothesis Generation.
"""

import dspy
from pydantic import BaseModel, Field

# We need to define Pydantic models for DSPy to output structured data
# Note: DSPy 2.5+ supports Pydantic output natively via TypedPredictor

class DataSummary(BaseModel):
    description: str = Field(..., description="Summary of dataset characteristics")
    columns: list[str] = Field(..., description="List of columns")
    target: str = Field(..., description="Target column name")
    problem_type: str = Field(..., description="Regression or Classification")

class GeneratedHypothesis(BaseModel):
    id: str = Field(..., description="Unique ID like H_01")
    statement: str = Field(..., description="The hypothesis statement")
    rationale: str = Field(..., description="Why this hypothesis is likely true based on data")
    experiment_design: str = Field(..., description="High level plan to test it")

class HypothesisList(BaseModel):
    hypotheses: list[GeneratedHypothesis]

class GenerateHypothesesSignature(dspy.Signature):
    """
    Given a dataset summary, generate a list of scientific hypotheses to improve model performance.
    Focus on feature engineering, preprocessing, and model architecture.
    """
    data_summary: DataSummary = dspy.InputField(desc="Characteristics of the training data")
    hypotheses: HypothesisList = dspy.OutputField(desc="List of testable hypotheses")

class HypothesisGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought for reasoning before outputting the list
        self.prog = dspy.ChainOfThought(GenerateHypothesesSignature)
        
    def forward(self, data_summary: DataSummary) -> HypothesisList:
        return self.prog(data_summary=data_summary).hypotheses
