"""LangGraph state definition for the agent swarm."""

from datetime import datetime
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from comet_swarm.schemas import (
    AgentMessage,
    BudgetStatus,
    CodeArtifact,
    CompetitionContext,
    DatasetInfo,
    ExecutionResult,
    ExperimentResult,
    StrategyProposal,
)


def reduce_experiments(
    existing: list[ExperimentResult],
    new: list[ExperimentResult],
) -> list[ExperimentResult]:
    """Reducer to append new experiments to existing list."""
    return existing + new


def reduce_artifacts(
    existing: list[CodeArtifact],
    new: list[CodeArtifact],
) -> list[CodeArtifact]:
    """Reducer to append new code artifacts."""
    return existing + new


class SwarmState(BaseModel):
    """
    State object that flows through the LangGraph workflow.
    
    This state is shared across all agent nodes and updated
    as the workflow progresses.
    """
    
    # ==========================================================================
    # Competition Context (Set once at start)
    # ==========================================================================
    competition: CompetitionContext | None = None
    
    # ==========================================================================
    # Data Information
    # ==========================================================================
    dataset_info: DatasetInfo | None = None
    eda_insights: dict[str, Any] = Field(default_factory=dict)
    
    # ==========================================================================
    # Strategy
    # ==========================================================================
    current_strategy: StrategyProposal | None = None
    strategy_iteration: int = 0
    
    # ==========================================================================
    # Code Artifacts (Appendable)
    # ==========================================================================
    code_artifacts: Annotated[list[CodeArtifact], reduce_artifacts] = Field(
        default_factory=list
    )
    current_code: str | None = None
    
    # ==========================================================================
    # Execution
    # ==========================================================================
    last_execution: ExecutionResult | None = None
    retry_count: int = 0
    max_retries: int = 3
    
    # ==========================================================================
    # Experiments (Appendable)
    # ==========================================================================
    experiments: Annotated[list[ExperimentResult], reduce_experiments] = Field(
        default_factory=list
    )
    best_score: float | None = None
    
    # ==========================================================================
    # Messages (Standard LangGraph message handling)
    # ==========================================================================
    messages: Annotated[list[AgentMessage], add_messages] = Field(default_factory=list)
    
    # ==========================================================================
    # Budget Tracking
    # ==========================================================================
    budget_status: BudgetStatus | None = None
    
    # ==========================================================================
    # Control Flow
    # ==========================================================================
    current_phase: str = "init"  # init, eda, strategy, code, execute, evaluate, submit
    iteration: int = 0
    max_iterations: int = 10
    should_stop: bool = False
    stop_reason: str | None = None
    
    # ==========================================================================
    # Timestamps
    # ==========================================================================
    started_at: datetime = Field(default_factory=datetime.now)
    last_updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

    def update_best_score(self, score: float, metric_direction: str = "minimize") -> bool:
        """
        Update best score if new score is better.
        
        Args:
            score: New score to compare
            metric_direction: 'minimize' or 'maximize'
        
        Returns:
            True if best score was updated
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        is_better = (
            (metric_direction == "minimize" and score < self.best_score) or
            (metric_direction == "maximize" and score > self.best_score)
        )
        
        if is_better:
            self.best_score = score
            return True
        
        return False

    def can_continue(self) -> bool:
        """Check if workflow can continue."""
        if self.should_stop:
            return False
        if self.iteration >= self.max_iterations:
            return False
        if self.budget_status and self.budget_status.is_exceeded:
            return False
        return True

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of current state for logging."""
        return {
            "phase": self.current_phase,
            "iteration": self.iteration,
            "experiments_count": len(self.experiments),
            "best_score": self.best_score,
            "strategy_iteration": self.strategy_iteration,
            "retry_count": self.retry_count,
            "budget_used": (
                self.budget_status.percentage_used if self.budget_status else 0
            ),
        }
