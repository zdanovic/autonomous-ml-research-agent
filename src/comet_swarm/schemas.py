"""Core Pydantic schemas for Comet-Swarm."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class Platform(str, Enum):
    """Supported competition platforms."""

    WUNDERNN = "wundernn"
    SOLAFUNE = "solafune"
    KAGGLE = "kaggle"


class MetricType(str, Enum):
    """Competition metric types."""

    PEARSON = "pearson"  # Wundernn
    RMSLE = "rmsle"  # Solafune
    R2 = "r2"  # Kaggle Flood
    RMSE = "rmse"
    MAE = "mae"
    AUC = "auc"
    F1 = "f1"
    ACCURACY = "accuracy"


class AgentRole(str, Enum):
    """Agent roles in the swarm."""

    STRATEGIST = "strategist"
    CODER = "coder"
    EVALUATOR = "evaluator"
    REVIEWER = "reviewer"


class ExecutionStatus(str, Enum):
    """Code execution status."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"


class ErrorType(str, Enum):
    """Error classification for self-correction."""

    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGIC = "logic"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    DATA_SCHEMA = "data_schema"
    UNKNOWN = "unknown"


# =============================================================================
# Competition Context
# =============================================================================


class CompetitionContext(BaseModel):
    """Competition metadata and configuration."""

    name: str = Field(..., description="Competition name")
    platform: Platform
    description: str = Field(..., description="Competition description/overview")
    metric: MetricType
    metric_direction: str = Field(default="minimize", pattern="^(minimize|maximize)$")
    deadline: datetime | None = None
    prize: str | None = None
    
    # Data info
    train_path: str | None = None
    test_path: str | None = None
    target_column: str | None = None
    
    # Platform-specific config
    submission_format: str = Field(default="csv", description="csv, parquet, or python")
    api_endpoint: str | None = None


# =============================================================================
# Dataset Info
# =============================================================================


class ColumnInfo(BaseModel):
    """Information about a single column."""

    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: list[Any] = Field(default_factory=list)
    
    # Numeric stats (optional)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    
    # Categorical stats (optional)
    top_categories: dict[str, int] | None = None


class DatasetInfo(BaseModel):
    """Dataset schema and statistics."""

    name: str
    num_rows: int
    num_columns: int
    memory_mb: float
    columns: list[ColumnInfo]
    
    # Target analysis
    target_column: str | None = None
    target_distribution: dict[str, Any] | None = None


# =============================================================================
# Agent Messages
# =============================================================================


class AgentMessage(BaseModel):
    """Message passed between agents."""

    role: AgentRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyProposal(BaseModel):
    """ML strategy proposed by Strategist agent."""

    hypotheses: list[str] = Field(..., description="Feature engineering hypotheses")
    model_recommendations: list[str] = Field(..., description="Recommended models")
    priority: str = Field(..., description="What to focus on first")
    estimated_impact: str | None = None


class CodeArtifact(BaseModel):
    """Generated code from Coder agent."""

    code: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    expected_output: str | None = None


# =============================================================================
# Execution Results
# =============================================================================


class ExecutionResult(BaseModel):
    """Result from sandbox code execution."""

    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    
    # Artifacts produced
    artifacts: dict[str, Any] = Field(default_factory=dict)
    
    # Error details (if status is ERROR)
    error_type: ErrorType | None = None
    error_message: str | None = None
    error_traceback: str | None = None


class ExperimentResult(BaseModel):
    """ML experiment result logged to Comet."""

    experiment_id: str
    experiment_url: str
    model_type: str
    hyperparameters: dict[str, Any]
    
    # Metrics
    train_score: float | None = None
    val_score: float | None = None
    cv_scores: list[float] = Field(default_factory=list)
    cv_mean: float | None = None
    cv_std: float | None = None
    
    # Feature importance
    feature_importance: dict[str, float] = Field(default_factory=dict)
    
    # Training curves
    training_history: dict[str, list[float]] = Field(default_factory=dict)


# =============================================================================
# Submission
# =============================================================================


class SubmissionResult(BaseModel):
    """Result from platform submission."""

    submission_id: str | None = None
    status: str  # pending, scored, failed
    public_score: float | None = None
    private_score: float | None = None
    message: str | None = None
    submitted_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Token Tracking
# =============================================================================


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    task_type: str
    timestamp: datetime = Field(default_factory=datetime.now)


class BudgetStatus(BaseModel):
    """Current token budget status."""

    total_spent_usd: float
    max_budget_usd: float
    remaining_usd: float
    usage_history: list[TokenUsage] = Field(default_factory=list)
    
    @property
    def is_exceeded(self) -> bool:
        return self.total_spent_usd >= self.max_budget_usd
    
    @property
    def percentage_used(self) -> float:
        return (self.total_spent_usd / self.max_budget_usd) * 100
