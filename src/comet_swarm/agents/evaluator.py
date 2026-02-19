"""Evaluator agent for analyzing experiment results."""

from typing import Any

from comet_swarm.agents.base import BaseAgent
from comet_swarm.schemas import AgentRole


class EvaluatorAgent(BaseAgent):
    """
    Evaluator agent that analyzes experiment results and suggests improvements.
    
    Responsibilities:
    - Analyze cross-validation results
    - Compare model performance
    - Identify overfitting/underfitting
    - Suggest hyperparameter tuning directions
    - Decide when to submit to leaderboard
    """

    role = AgentRole.EVALUATOR

    def __init__(self, model: str | None = None):
        super().__init__(
            model=model,
            temperature=0.0,
            task_type="error_analysis",
        )

    def get_system_prompt(self, context: dict[str, Any]) -> str:
        """Build system prompt with experiment context."""
        metric = context.get("metric", "unknown")
        metric_direction = context.get("metric_direction", "minimize")
        best_score = context.get("best_score", "N/A")
        experiments = context.get("experiments", [])
        
        experiments_table = ""
        if experiments:
            experiments_table = "\n".join([
                f"| {e.get('name', 'exp')} | {e.get('model_type', 'unknown')} | "
                f"{e.get('cv_mean', 'N/A'):.4f} | {e.get('cv_std', 'N/A'):.4f} |"
                for e in experiments[-10:]
            ])
            experiments_table = (
                "| Name | Model | CV Mean | CV Std |\n"
                "|------|-------|---------|--------|\n"
                + experiments_table
            )
        
        return f"""You are an ML evaluation expert analyzing competition experiments.

## Competition Context
- **Metric**: {metric} ({metric_direction})
- **Best Score So Far**: {best_score}

## Experiment History
{experiments_table if experiments_table else "No experiments yet."}

## Your Analysis Tasks
1. **Performance Analysis**: What's working? What's not?
2. **Overfitting Detection**: Is there a train-val gap?
3. **Feature Impact**: Which features drive performance?
4. **Next Steps**: What should we try next?
5. **Submit Decision**: Should we submit to the leaderboard?

## Output Format
Provide structured analysis in JSON:

```json
{{
    "performance_summary": "Brief summary of current performance",
    "overfitting_status": "none|mild|severe",
    "key_insights": ["insight 1", "insight 2"],
    "recommended_actions": ["action 1", "action 2"],
    "should_submit": true/false,
    "submit_reason": "Why or why not to submit"
}}
```"""

    async def analyze_results(
        self,
        experiment_results: list[dict[str, Any]],
        metric: str,
        metric_direction: str = "minimize",
        current_leaderboard_score: float | None = None,
    ) -> dict[str, Any]:
        """
        Analyze experiment results and provide recommendations.
        
        Args:
            experiment_results: List of experiment result dictionaries
            metric: Competition metric name
            metric_direction: 'minimize' or 'maximize'
            current_leaderboard_score: Current public LB score if any
        
        Returns:
            Analysis dictionary with insights and recommendations
        """
        # Find best score
        if experiment_results:
            scores = [e.get("cv_mean") for e in experiment_results if e.get("cv_mean")]
            if scores:
                best_score = min(scores) if metric_direction == "minimize" else max(scores)
            else:
                best_score = None
        else:
            best_score = None
        
        context = {
            "metric": metric,
            "metric_direction": metric_direction,
            "best_score": best_score,
            "experiments": experiment_results,
        }
        
        user_message = f"""Analyze the experiment results above.

Current leaderboard score: {current_leaderboard_score if current_leaderboard_score else 'No submission yet'}

Provide:
1. What patterns do you see in the results?
2. Are we overfitting?
3. What should we try next?
4. Should we submit our best model?"""

        response = await self.invoke(user_message, context)
        
        # Parse response
        import json
        import re
        
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Fallback
        return {
            "performance_summary": "Unable to parse analysis",
            "overfitting_status": "unknown",
            "key_insights": [],
            "recommended_actions": ["Review raw response"],
            "should_submit": False,
            "submit_reason": "Analysis failed",
        }

    async def compare_models(
        self,
        model_a: dict[str, Any],
        model_b: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compare two models and recommend which to use.
        
        Args:
            model_a: First model's results
            model_b: Second model's results
        
        Returns:
            Comparison analysis with recommendation
        """
        user_message = f"""Compare these two models:

**Model A** ({model_a.get('model_type', 'unknown')}):
- CV Mean: {model_a.get('cv_mean', 'N/A')}
- CV Std: {model_a.get('cv_std', 'N/A')}
- Features: {len(model_a.get('feature_importance', {}))}

**Model B** ({model_b.get('model_type', 'unknown')}):
- CV Mean: {model_b.get('cv_mean', 'N/A')}
- CV Std: {model_b.get('cv_std', 'N/A')}
- Features: {len(model_b.get('feature_importance', {}))}

Which model is better and why? Consider:
1. Raw performance
2. Stability (lower std is better)
3. Complexity (simpler is often better)

Respond in JSON with 'winner', 'reason', and 'ensemble_potential'."""

        response = await self.invoke(user_message, {})
        
        import json
        import re
        
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        return {"winner": "model_a", "reason": "Unable to parse", "ensemble_potential": False}
