"""Strategist agent for ML competition analysis and strategy planning."""

from typing import Any

from comet_swarm.agents.base import BaseAgent
from comet_swarm.schemas import AgentRole, StrategyProposal


class StrategistAgent(BaseAgent):
    """
    Strategist agent that analyzes competition data and proposes ML strategies.
    
    Responsibilities:
    - Analyze competition description and objectives
    - Review EDA insights
    - Propose feature engineering hypotheses
    - Recommend model architectures
    - Prioritize experiments
    """

    role = AgentRole.STRATEGIST

    def __init__(self, model: str | None = None):
        super().__init__(
            model=model,
            temperature=0.1,  # Slight creativity for strategy
            task_type="strategy_generation",
        )

    def get_system_prompt(self, context: dict[str, Any]) -> str:
        """Build system prompt with competition context."""
        competition_name = context.get("competition_name", "Unknown Competition")
        metric = context.get("metric", "unknown")
        metric_direction = context.get("metric_direction", "minimize")
        target_column = context.get("target_column", "target")
        dataset_info = context.get("dataset_info", "No dataset info available")
        eda_insights = context.get("eda_insights", "No EDA insights available")
        previous_experiments = context.get("previous_experiments", [])
        
        experiments_summary = ""
        if previous_experiments:
            experiments_summary = "\n".join([
                f"- {exp.get('model_type')}: {exp.get('cv_mean', 'N/A')} (±{exp.get('cv_std', 'N/A')})"
                for exp in previous_experiments[-5:]  # Last 5
            ])
        else:
            experiments_summary = "No experiments run yet."
        
        return f"""You are a Kaggle Grandmaster strategist specializing in ML competitions.

## Competition Context
- **Name**: {competition_name}
- **Metric**: {metric} ({metric_direction})
- **Target**: {target_column}

## Dataset Overview
{dataset_info}

## EDA Insights
{eda_insights}

## Previous Experiments
{experiments_summary}

## Your Task
Analyze the competition and propose a winning strategy. You must:

1. **Identify Key Patterns**: What are the most important signals in the data?
2. **Feature Engineering Hypotheses**: Propose 3-5 specific feature ideas with rationale.
3. **Model Recommendations**: Suggest which models to try and why.
4. **Priority Order**: What should we focus on first?

## Output Format
Respond with a structured strategy in this format:

```json
{{
    "hypotheses": [
        "Hypothesis 1: Description and rationale",
        "Hypothesis 2: Description and rationale",
        ...
    ],
    "model_recommendations": [
        "Model 1: Why this model fits the problem",
        "Model 2: Why this model fits the problem",
        ...
    ],
    "priority": "What to focus on first and why",
    "estimated_impact": "Expected improvement from this strategy"
}}
```

Be specific, data-driven, and actionable. Avoid generic advice."""

    async def generate_strategy(
        self,
        competition_context: dict[str, Any],
        dataset_info: str,
        eda_insights: str,
        previous_experiments: list[dict[str, Any]] | None = None,
    ) -> StrategyProposal:
        """
        Generate ML strategy based on competition analysis.
        
        Args:
            competition_context: Competition metadata
            dataset_info: Formatted dataset information
            eda_insights: Key EDA findings
            previous_experiments: List of previous experiment results
        
        Returns:
            StrategyProposal with hypotheses and recommendations
        """
        context = {
            "competition_name": competition_context.get("name", "Competition"),
            "metric": competition_context.get("metric", "unknown"),
            "metric_direction": competition_context.get("metric_direction", "minimize"),
            "target_column": competition_context.get("target_column", "target"),
            "dataset_info": dataset_info,
            "eda_insights": eda_insights,
            "previous_experiments": previous_experiments or [],
        }
        
        user_message = """Based on the competition context and EDA insights, generate a winning strategy.
Focus on:
1. What unique patterns can we exploit?
2. What features would capture the signal?
3. Which models are best suited for this problem?
4. What's the most impactful thing to try first?"""
        
        response = await self.invoke(user_message, context)
        
        # Parse response
        import json
        import re
        
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return StrategyProposal(
                    hypotheses=data.get("hypotheses", []),
                    model_recommendations=data.get("model_recommendations", []),
                    priority=data.get("priority", "Start with baseline"),
                    estimated_impact=data.get("estimated_impact"),
                )
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract from text
        return StrategyProposal(
            hypotheses=["Unable to parse - review raw response"],
            model_recommendations=["LightGBM baseline"],
            priority="Start with baseline model",
            estimated_impact=None,
        )

    async def refine_strategy(
        self,
        current_strategy: StrategyProposal,
        experiment_results: list[dict[str, Any]],
        feedback: str | None = None,
    ) -> StrategyProposal:
        """
        Refine strategy based on experiment results.
        
        Args:
            current_strategy: Current strategy being executed
            experiment_results: Results from recent experiments
            feedback: Optional human feedback
        
        Returns:
            Refined StrategyProposal
        """
        context = {
            "current_strategy": current_strategy.model_dump(),
            "experiment_results": experiment_results,
        }
        
        user_message = f"""Our current strategy was:
- Hypotheses: {current_strategy.hypotheses}
- Priority: {current_strategy.priority}

Experiment results:
{experiment_results}

{f'Human feedback: {feedback}' if feedback else ''}

Based on these results, how should we refine our strategy? What worked? What didn't?
Generate an updated strategy."""

        response = await self.invoke(user_message, context)
        
        # Parse and return
        import json
        import re
        
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return StrategyProposal(
                    hypotheses=data.get("hypotheses", current_strategy.hypotheses),
                    model_recommendations=data.get(
                        "model_recommendations", current_strategy.model_recommendations
                    ),
                    priority=data.get("priority", "Continue current approach"),
                    estimated_impact=data.get("estimated_impact"),
                )
            except json.JSONDecodeError:
                pass
        
        return current_strategy
