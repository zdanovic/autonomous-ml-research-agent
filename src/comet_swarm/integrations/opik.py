"""Comet Opik integration for LLM observability."""

from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import opik
from opik import Opik, track
from opik.evaluation.metrics import base_metric

from comet_swarm.config import get_settings
from comet_swarm.schemas import AgentRole, BudgetStatus, TokenUsage

P = ParamSpec("P")
R = TypeVar("R")


class TokenBudgetManager:
    """Manages token budget across all LLM calls."""

    def __init__(self, max_budget_usd: float = 10.0):
        self.max_budget = max_budget_usd
        self._usage_history: list[TokenUsage] = []
        self._total_spent: float = 0.0

    def log_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str,
    ) -> TokenUsage:
        """Log token usage and calculate cost."""
        settings = get_settings()
        costs = settings.get_token_cost(model)
        
        cost = (
            (input_tokens * costs["input"] / 1_000_000)
            + (output_tokens * costs["output"] / 1_000_000)
        )
        
        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_type=task_type,
        )
        
        self._usage_history.append(usage)
        self._total_spent += cost
        
        return usage

    def get_status(self) -> BudgetStatus:
        """Get current budget status."""
        return BudgetStatus(
            total_spent_usd=self._total_spent,
            max_budget_usd=self.max_budget,
            remaining_usd=max(0, self.max_budget - self._total_spent),
            usage_history=self._usage_history,
        )

    def can_proceed(self) -> bool:
        """Check if we can make more LLM calls."""
        return self._total_spent < self.max_budget

    def get_recommended_tier(self) -> str:
        """Get recommended model tier based on remaining budget."""
        status = self.get_status()
        
        if status.percentage_used > 80:
            return "fast"
        elif status.percentage_used > 50:
            return "standard"
        else:
            return "premium"


class OpikTracer:
    """Wrapper for Opik tracing with budget management."""

    def __init__(self, project_name: str | None = None):
        settings = get_settings()
        
        # Configure Opik (project set via environment or client)
        opik.configure(
            api_key=settings.effective_opik_key,
            force=True,  # Use new settings
        )
        
        self.project_name = project_name or settings.opik_project_name
        self.client = Opik()
        self.budget_manager = TokenBudgetManager(settings.max_budget_usd)
        self._current_trace_id: str | None = None

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start a new trace."""
        trace = self.client.trace(
            name=name,
            metadata=metadata or {},
        )
        self._current_trace_id = trace.id
        return trace.id

    def log_span(
        self,
        name: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a span within the current trace."""
        if self._current_trace_id:
            self.client.span(
                trace_id=self._current_trace_id,
                name=name,
                input=input_data,
                output=output_data,
                metadata=metadata or {},
            )

    def log_agent_step(
        self,
        agent_role: AgentRole,
        input_prompt: str,
        output_response: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str,
    ) -> TokenUsage:
        """Log an agent step with token tracking."""
        # Track tokens
        usage = self.budget_manager.log_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            task_type=task_type,
        )
        
        # Log to Opik
        self.log_span(
            name=f"{agent_role.value}_step",
            input_data={"prompt": input_prompt[:500]},  # Truncate for display
            output_data={"response": output_response[:500]},
            metadata={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": usage.cost_usd,
                "task_type": task_type,
                "budget_remaining": self.budget_manager.get_status().remaining_usd,
            },
        )
        
        return usage

    def get_budget_status(self) -> BudgetStatus:
        """Get current token budget status."""
        return self.budget_manager.get_status()


def traced(
    name: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for tracing function calls with Opik."""
    
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        @track(
            name=name or func.__name__,
            capture_input=capture_input,
            capture_output=capture_output,
        )
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# =============================================================================
# Custom Evaluation Metrics
# =============================================================================


class StrategyCoherenceMetric(base_metric.BaseMetric):
    """Evaluates logical coherence of ML strategy proposals."""

    name = "strategy_coherence"

    def score(
        self,
        output: str,
        competition_description: str | None = None,
        **kwargs: Any,
    ) -> float:
        """Score strategy coherence (0-1)."""
        # Simple heuristic scoring
        score = 0.0
        
        # Check for key strategy components
        if any(word in output.lower() for word in ["baseline", "benchmark"]):
            score += 0.2
        if any(word in output.lower() for word in ["feature", "engineering"]):
            score += 0.2
        if any(word in output.lower() for word in ["cross-validation", "cv", "fold"]):
            score += 0.2
        if any(word in output.lower() for word in ["ensemble", "stacking", "blend"]):
            score += 0.2
        if any(word in output.lower() for word in ["hypothesis", "experiment"]):
            score += 0.2
        
        return min(1.0, score)


class CodeExecutabilityMetric(base_metric.BaseMetric):
    """Tracks code execution success rate."""

    name = "code_executability"

    def score(
        self,
        execution_status: str,
        **kwargs: Any,
    ) -> float:
        """Score based on execution status."""
        status_scores = {
            "success": 1.0,
            "error_fixed": 0.5,
            "error": 0.0,
            "timeout": 0.0,
            "oom": 0.0,
        }
        return status_scores.get(execution_status, 0.0)


class InsightActionabilityMetric(base_metric.BaseMetric):
    """Measures if EDA insights lead to score improvements."""

    name = "insight_actionability"

    def score(
        self,
        insights_count: int,
        score_delta: float,
        **kwargs: Any,
    ) -> float:
        """Score based on insight impact."""
        if insights_count == 0:
            return 0.0
        
        # Average impact per insight
        impact_per_insight = score_delta / insights_count
        
        # Normalize (assume good insight gives +0.01 to score)
        return min(1.0, max(0.0, impact_per_insight / 0.01))


# Singleton tracer instance
_tracer: OpikTracer | None = None


def get_tracer() -> OpikTracer:
    """Get or create singleton tracer."""
    global _tracer
    if _tracer is None:
        _tracer = OpikTracer()
    return _tracer
