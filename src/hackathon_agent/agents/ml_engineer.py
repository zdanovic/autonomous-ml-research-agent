"""
ML Engineer Agent.

Responsibility: Execute experiments and train models.
"""
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..core.agent import Agent, AgentState
from ..schema.experiment import ExperimentResult, ExperimentSpec, ExperimentStatus


class MLEngineerState(AgentState):
    results: list[ExperimentResult] = []

class MLEngineer(Agent[MLEngineerState]):
    """
    Agent that executes ML experiments.
    """
    
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute planned experiments.
        """
        experiments: list[ExperimentSpec] = context.get("planned_experiments", [])
        
        train_df = context.get("train_df")
        target_col = context.get("target_col")
        
        for exp in experiments:
            if exp.status == ExperimentStatus.PLANNED:
                self.log(f"Running experiment {exp.id}: {exp.name}")
                exp.status = ExperimentStatus.RUNNING
                
                try:
                    result = self._run_experiment(exp, train_df, target_col)
                    self.state.results.append(result)
                    exp.status = ExperimentStatus.COMPLETED
                    self.log(f"Experiment {exp.id} finished. CV: {result.cv_score:.4f}")
                except Exception as e:
                    self.log(f"Experiment {exp.id} failed: {e}", level="ERROR")
                    exp.status = ExperimentStatus.FAILED
                    
        return {
            "results": self.state.results
        }

    def _run_experiment(self, exp: ExperimentSpec, df: pd.DataFrame, target_col: str) -> ExperimentResult:
        """
        Execute a single experiment.
        """
        # Simple CV loop for now (Strategy pattern to be expanded)
        if exp.strategy == "catboost_grouped":
            return self._run_catboost(exp, df, target_col)
        
        # Fallback/Generic
        return self._run_generic(exp, df, target_col)

    def _run_generic(self, exp: ExperimentSpec, df: pd.DataFrame, target_col: str) -> ExperimentResult:
        """Generic CV run."""
        from catboost import CatBoostRegressor
        
        start_time = datetime.now()
        
        kf = KFold(n_splits=exp.n_folds, shuffle=True, random_state=42)
        X = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=[np.number])
        y = df[target_col]
        
        oof = np.zeros(len(df))
        
        for tr_idx, val_idx in kf.split(X, y):
            model = CatBoostRegressor(verbose=0, allow_writing_files=False, iterations=100)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[val_idx] = model.predict(X.iloc[val_idx])
            
        cv_score = np.sqrt(np.mean((np.log1p(oof) - np.log1p(y))**2)) # RMSLE approximation
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return ExperimentResult(
            experiment_id=exp.id,
            cv_score=float(cv_score),
            duration_seconds=duration,
            success=True
        )

    def _run_catboost(self, exp: ExperimentSpec, df: pd.DataFrame, target_col: str) -> ExperimentResult:
        """CatBoost run."""
        # TODO: Implement proper strategy dispatch
        return self._run_generic(exp, df, target_col)
