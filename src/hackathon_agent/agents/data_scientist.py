"""
Data Scientist Agent - Refactored.

Responsibility: Analyze data and generate testable hypotheses.
"""
from typing import Any

import pandas as pd

from ..brain.hypothesis_generation import DataSummary
from ..core.agent import Agent, AgentState
from ..schema.experiment import Hypothesis, HypothesisPriority


class DataScientistState(AgentState):
    hypotheses: list[Hypothesis] = []
    insights: list[dict[str, Any]] = []
    last_data_summary: DataSummary | None = None

class DataScientist(Agent[DataScientistState]):
    """
    Agent that performs EDA and generates hypotheses.
    """
    
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze dataset and produce hypotheses.
        """
        self.log("Starting deep data analysis...")
        
        train_df = context.get("train_df")
        if train_df is None:
            raise ValueError("Train dataframe not found in context")
            
        target_col = context.get("target_col")
        
        # 1. Generate Hypotheses based on data properties
        new_hypotheses = self._generate_hypotheses(train_df, target_col)
        
        self.state.hypotheses.extend(new_hypotheses)
        self.log(f"Generated {len(new_hypotheses)} hypotheses.")
        
        return {
            "hypotheses": self.state.hypotheses,
            "insights": self.state.insights
        }

    def _generate_hypotheses(self, df: pd.DataFrame, target_col: str) -> list[Hypothesis]:
        """
        Generate hypotheses using DSPy module if available, fall back to heuristics.
        """
        hypotheses = []
        
        # Try DSPy if Config is available
        try:
            from ..brain.config import init_dspy
            from ..brain.hypothesis_generation import DataSummary, HypothesisGenerator
            
            init_dspy()
            
            # Create summary for LLM
            summary = DataSummary(
                description=f"DataFrame with {len(df)} rows and {len(df.columns)} columns.",
                columns=list(df.columns),
                target=target_col,
                problem_type="Regression" if pd.api.types.is_numeric_dtype(df[target_col]) else "Classification"
            )
            
            # Save summary to state for later optimization usage
            self.state.last_data_summary = summary
            
            # Load optimized program if available
            opt_path = "src/hackathon_agent/brain/compiled_hypothesizer.json"
            generator = HypothesisGenerator()
            if os.path.exists(opt_path):
                try:
                    generator.load(opt_path)
                    self.log("Loaded OPTIMIZED DSPy program for hypothesis generation.")
                except Exception as e:
                    self.log(f"Failed to load optimized program: {e}. Using default.")
            
            dspy_hypotheses = generator(data_summary=summary)
            
            # Convert DSPy output to internal Hypothesis schema
            for h in dspy_hypotheses:
                hypotheses.append(Hypothesis(
                    id=h.id,
                    statement=h.statement,
                    rationale=h.rationale,
                    experiment_design=h.experiment_design,
                    expected_improvement=0.01, # Placeholder
                    priority=HypothesisPriority.HIGH
                ))
                
            if hypotheses:
                return hypotheses

        except Exception as e:
            self.log(f"DSPy generation failed: {e}. Falling back to heuristics.", level="WARNING")
        
        # Fallback Heuristics
        # 1. Check for high cardinality
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if df[col].nunique() > 50:
                hypotheses.append(Hypothesis(
                    id="H_TARGET_ENC",
                    statement=f"Target encoding on {col} will improve performance",
                    rationale=f"High cardinality ({df[col].nunique()}) feature detected.",
                    experiment_design="Apply smoothed target encoding",
                    expected_improvement=0.015,
                    priority=HypothesisPriority.HIGH
                ))
                break # Avoid duplicates for now
        
        # 2. Check for skewed target
        if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
            skew = df[target_col].skew()
            if abs(skew) > 1.0:
                 hypotheses.append(Hypothesis(
                    id="H_LOG_TARGET",
                    statement="Log-transform target variable",
                    rationale=f"Target is skewed ({skew:.2f}). Log-transform normalizes distribution.",
                    experiment_design="Train on log1p(target), predict expm1(pred)",
                    expected_improvement=0.02,
                    priority=HypothesisPriority.CRITICAL
                ))
        
        # 3. Country Specific (Domain Knowledge)
        if "country" in df.columns:
             hypotheses.append(Hypothesis(
                id="H_COUNTRY_SPECIFIC",
                statement="Train separate models per country",
                rationale="Data contains 'country' column. Local patterns likely vary.",
                experiment_design="Split data by country, train independent models",
                expected_improvement=0.03,
                priority=HypothesisPriority.CRITICAL
            ))

        return hypotheses
