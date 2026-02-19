"""LangGraph workflow orchestration for the agent swarm."""

from typing import Literal

from langgraph.graph import END, StateGraph

from comet_swarm.agents import CoderAgent, EvaluatorAgent, ReviewerAgent, StrategistAgent
from comet_swarm.graph.state import SwarmState
from comet_swarm.integrations.opik import get_tracer
from comet_swarm.schemas import (
    AgentMessage,
    AgentRole,
    ExecutionResult,
    ExecutionStatus,
)

# =============================================================================
# Node Functions
# =============================================================================


async def eda_node(state: SwarmState) -> dict:
    """
    Run EDA on the dataset.
    
    This node profiles the dataset and generates initial insights.
    """
    import polars as pl

    from comet_swarm.tools.eda import correlation_analysis, distribution_analysis, profile_dataset
    
    tracer = get_tracer()
    tracer.start_trace("eda_phase")
    
    updates = {
        "current_phase": "eda",
        "messages": [],
    }
    
    try:
        if state.competition and state.competition.train_path:
            # Load and profile dataset
            df = pl.read_parquet(state.competition.train_path)
            
            # Profile
            dataset_info = profile_dataset(
                df,
                name=state.competition.name,
                target_column=state.competition.target_column,
            )
            updates["dataset_info"] = dataset_info
            
            # Correlations
            if state.competition.target_column:
                correlations = correlation_analysis(
                    df,
                    state.competition.target_column,
                    top_n=15,
                )
                updates["eda_insights"] = {
                    "top_correlations": correlations,
                    "distributions": distribution_analysis(df),
                }
            
            updates["messages"] = [
                AgentMessage(
                    role=AgentRole.CODER,
                    content=f"EDA completed. Found {len(dataset_info.columns)} columns, "
                            f"{dataset_info.num_rows} rows.",
                )
            ]
    except Exception as e:
        updates["messages"] = [
            AgentMessage(
                role=AgentRole.REVIEWER,
                content=f"EDA failed: {e!s}",
            )
        ]
    
    return updates


async def strategy_node(state: SwarmState) -> dict:
    """
    Generate or refine ML strategy.
    """
    strategist = StrategistAgent()
    
    updates = {
        "current_phase": "strategy",
        "messages": [],
    }
    
    # Format dataset info
    dataset_summary = "No dataset info available."
    if state.dataset_info:
        columns_info = ", ".join([
            f"{c.name} ({c.dtype})" 
            for c in state.dataset_info.columns[:10]
        ])
        dataset_summary = (
            f"Rows: {state.dataset_info.num_rows}, "
            f"Columns: {state.dataset_info.num_columns}\n"
            f"Sample columns: {columns_info}"
        )
    
    # Format EDA insights
    eda_summary = "No EDA insights."
    if state.eda_insights:
        top_corrs = state.eda_insights.get("top_correlations", {})
        if top_corrs:
            corr_items = list(top_corrs.items())[:5]
            eda_summary = "Top correlations with target:\n" + "\n".join([
                f"- {k}: {v:.4f}" for k, v in corr_items
            ])
    
    # Generate strategy
    if state.current_strategy is None or state.strategy_iteration == 0:
        # Initial strategy
        competition_context = state.competition.model_dump() if state.competition else {}
        strategy = await strategist.generate_strategy(
            competition_context=competition_context,
            dataset_info=dataset_summary,
            eda_insights=eda_summary,
            previous_experiments=[e.model_dump() for e in state.experiments],
        )
    else:
        # Refine strategy
        strategy = await strategist.refine_strategy(
            current_strategy=state.current_strategy,
            experiment_results=[e.model_dump() for e in state.experiments[-3:]],
        )
    
    updates["current_strategy"] = strategy
    updates["strategy_iteration"] = state.strategy_iteration + 1
    updates["messages"] = [
        AgentMessage(
            role=AgentRole.STRATEGIST,
            content=f"Strategy generated: {strategy.priority}",
        )
    ]
    
    return updates


async def code_node(state: SwarmState) -> dict:
    """
    Generate code based on current strategy.
    """
    coder = CoderAgent()
    
    updates = {
        "current_phase": "code",
        "messages": [],
        "code_artifacts": [],
    }
    
    if not state.current_strategy:
        updates["messages"] = [
            AgentMessage(
                role=AgentRole.CODER,
                content="No strategy available. Cannot generate code.",
            )
        ]
        return updates
    
    # Get columns from dataset
    columns = []
    if state.dataset_info:
        columns = [c.name for c in state.dataset_info.columns]
    
    target = state.competition.target_column if state.competition else "target"
    
    # Generate code for first hypothesis
    hypothesis = state.current_strategy.hypotheses[0] if state.current_strategy.hypotheses else "Train baseline model"
    
    # Check what type of code to generate
    if state.iteration == 0 or not state.experiments:
        # Generate feature code first
        artifact = await coder.generate_feature_code(
            hypothesis=hypothesis,
            columns=columns,
            target_column=target,
        )
        updates["code_artifacts"] = [artifact]
        updates["current_code"] = artifact.code
    else:
        # Generate training code
        model_type = "lightgbm"
        if state.current_strategy.model_recommendations:
            rec = state.current_strategy.model_recommendations[0].lower()
            if "xgboost" in rec:
                model_type = "xgboost"
            elif "catboost" in rec:
                model_type = "catboost"
        
        artifact = await coder.generate_training_code(
            model_type=model_type,
            columns=columns,
            target_column=target,
            metric=state.competition.metric.value if state.competition else "rmse",
        )
        updates["code_artifacts"] = [artifact]
        updates["current_code"] = artifact.code
    
    updates["messages"] = [
        AgentMessage(
            role=AgentRole.CODER,
            content=f"Generated code: {artifact.description[:100]}",
        )
    ]
    
    return updates


async def execute_node(state: SwarmState) -> dict:
    """
    Execute code in sandbox.
    
    This is a placeholder - actual E2B execution would go here.
    """
    updates = {
        "current_phase": "execute",
        "messages": [],
    }
    
    if not state.current_code:
        updates["last_execution"] = ExecutionResult(
            status=ExecutionStatus.ERROR,
            error_message="No code to execute",
        )
        return updates
    
    # TODO: Replace with actual E2B execution
    # For now, simulate successful execution
    updates["last_execution"] = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        stdout="Execution completed successfully.",
        execution_time_seconds=30.5,
        memory_used_mb=512.0,
        artifacts={"model_path": "models/lightgbm_model.pkl"},
    )
    updates["retry_count"] = 0
    
    updates["messages"] = [
        AgentMessage(
            role=AgentRole.CODER,
            content="Code executed successfully.",
        )
    ]
    
    return updates


async def review_node(state: SwarmState) -> dict:
    """
    Review execution results and handle errors.
    """
    reviewer = ReviewerAgent()
    
    updates = {
        "current_phase": "review",
        "messages": [],
    }
    
    if not state.last_execution:
        return updates
    
    if state.last_execution.status == ExecutionStatus.SUCCESS:
        updates["messages"] = [
            AgentMessage(
                role=AgentRole.REVIEWER,
                content="Execution successful, proceeding to evaluation.",
            )
        ]
        return updates
    
    # Analyze error
    error_analysis = await reviewer.analyze_error(
        execution_result=state.last_execution,
        original_code=state.current_code or "",
    )
    
    # Decide retry
    should_retry, reason = await reviewer.should_retry(
        error_analysis=error_analysis,
        retry_count=state.retry_count,
        max_retries=state.max_retries,
    )
    
    if should_retry:
        updates["current_code"] = error_analysis.get("fixed_code")
        updates["retry_count"] = state.retry_count + 1
        updates["messages"] = [
            AgentMessage(
                role=AgentRole.REVIEWER,
                content=f"Retrying with fix: {error_analysis.get('fix_description')}",
            )
        ]
    else:
        updates["messages"] = [
            AgentMessage(
                role=AgentRole.REVIEWER,
                content=f"Cannot auto-fix: {reason}",
            )
        ]
    
    return updates


async def evaluate_node(state: SwarmState) -> dict:
    """
    Evaluate experiment results and decide next steps.
    """
    evaluator = EvaluatorAgent()
    
    updates = {
        "current_phase": "evaluate",
        "messages": [],
        "experiments": [],
    }
    
    # Check if we have results to evaluate
    if not state.last_execution or state.last_execution.status != ExecutionStatus.SUCCESS:
        return updates
    
    # Create experiment result (placeholder)
    from comet_swarm.schemas import ExperimentResult
    
    experiment = ExperimentResult(
        experiment_id=f"exp_{state.iteration}",
        experiment_url="https://www.comet.com/...",
        model_type="lightgbm",
        hyperparameters={},
        cv_mean=0.25,  # Placeholder
        cv_std=0.02,
    )
    updates["experiments"] = [experiment]
    
    # Analyze results
    all_experiments = state.experiments + [experiment]
    analysis = await evaluator.analyze_results(
        experiment_results=[e.model_dump() for e in all_experiments],
        metric=state.competition.metric.value if state.competition else "rmse",
        metric_direction=state.competition.metric_direction if state.competition else "minimize",
    )
    
    updates["messages"] = [
        AgentMessage(
            role=AgentRole.EVALUATOR,
            content=f"Analysis: {analysis.get('performance_summary', 'No summary')}",
        )
    ]
    
    # Update best score
    if experiment.cv_mean:
        metric_dir = state.competition.metric_direction if state.competition else "minimize"
        # Note: This requires special handling as state is immutable
        # In actual implementation, we'd return the new best_score
    
    # Check if should submit
    if analysis.get("should_submit", False):
        updates["current_phase"] = "submit"
    
    return updates


async def submit_node(state: SwarmState) -> dict:
    """
    Generate and submit predictions.
    """
    updates = {
        "current_phase": "submit",
        "messages": [],
    }
    
    # TODO: Implement actual submission
    updates["messages"] = [
        AgentMessage(
            role=AgentRole.CODER,
            content="Submission prepared. Ready to submit.",
        )
    ]
    
    return updates


async def end_node(state: SwarmState) -> dict:
    """
    Final node to wrap up the workflow.
    """
    return {
        "should_stop": True,
        "stop_reason": "Workflow completed",
        "messages": [
            AgentMessage(
                role=AgentRole.STRATEGIST,
                content=f"Workflow finished after {state.iteration} iterations. "
                        f"Best score: {state.best_score}",
            )
        ],
    }


# =============================================================================
# Router Functions
# =============================================================================


def route_after_eda(state: SwarmState) -> Literal["strategy", "end"]:
    """Route after EDA completes."""
    if state.dataset_info is None:
        return "end"
    return "strategy"


def route_after_execute(state: SwarmState) -> Literal["review", "evaluate"]:
    """Route based on execution result."""
    if state.last_execution and state.last_execution.status == ExecutionStatus.SUCCESS:
        return "evaluate"
    return "review"


def route_after_review(state: SwarmState) -> Literal["execute", "strategy", "end"]:
    """Route based on review decision."""
    if state.retry_count < state.max_retries and state.current_code:
        return "execute"
    if state.iteration < state.max_iterations:
        return "strategy"
    return "end"


def route_after_evaluate(state: SwarmState) -> Literal["submit", "strategy", "end"]:
    """Route based on evaluation."""
    if state.current_phase == "submit":
        return "submit"
    if not state.can_continue():
        return "end"
    return "strategy"


def route_after_submit(state: SwarmState) -> Literal["strategy", "end"]:
    """Route after submission."""
    if not state.can_continue():
        return "end"
    return "strategy"


# =============================================================================
# Workflow Builder
# =============================================================================


def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for the agent swarm.
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(SwarmState)
    
    # Add nodes
    workflow.add_node("eda", eda_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("code", code_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("review", review_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("submit", submit_node)
    workflow.add_node("end", end_node)
    
    # Set entry point
    workflow.set_entry_point("eda")
    
    # Add edges with routing
    workflow.add_conditional_edges("eda", route_after_eda)
    workflow.add_edge("strategy", "code")
    workflow.add_edge("code", "execute")
    workflow.add_conditional_edges("execute", route_after_execute)
    workflow.add_conditional_edges("review", route_after_review)
    workflow.add_conditional_edges("evaluate", route_after_evaluate)
    workflow.add_conditional_edges("submit", route_after_submit)
    workflow.add_edge("end", END)
    
    return workflow.compile()


async def run_workflow(
    competition: dict,
    max_iterations: int = 10,
) -> SwarmState:
    """
    Run the complete workflow.
    
    Args:
        competition: Competition configuration dict
        max_iterations: Maximum iterations
    
    Returns:
        Final SwarmState
    """
    from comet_swarm.schemas import CompetitionContext, MetricType, Platform
    
    # Create competition context
    context = CompetitionContext(
        name=competition.get("name", "Unknown"),
        platform=Platform(competition.get("platform", "kaggle")),
        description=competition.get("description", ""),
        metric=MetricType(competition.get("metric", "rmse")),
        metric_direction=competition.get("metric_direction", "minimize"),
        train_path=competition.get("train_path"),
        test_path=competition.get("test_path"),
        target_column=competition.get("target_column"),
        submission_format=competition.get("submission_format", "csv"),
    )
    
    # Initialize state
    initial_state = SwarmState(
        competition=context,
        max_iterations=max_iterations,
    )
    
    # Create and run workflow
    workflow = create_workflow()
    final_state = await workflow.ainvoke(initial_state)
    
    return final_state
