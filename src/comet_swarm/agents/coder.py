"""Coder agent for generating feature engineering and training code."""

from typing import Any

from comet_swarm.agents.base import BaseAgent, ToolCallMixin
from comet_swarm.schemas import AgentRole, CodeArtifact

# Available tools for the coder agent
AVAILABLE_TOOLS = [
    {
        "name": "target_encode",
        "description": "Apply target encoding with k-fold CV to prevent leakage",
        "usage": "target_encode(df, column='category', target='label', n_folds=5)",
    },
    {
        "name": "frequency_encode",
        "description": "Encode categorical column by frequency counts",
        "usage": "frequency_encode(df, column='category', normalize=True)",
    },
    {
        "name": "lag_features",
        "description": "Create lag features for time-series",
        "usage": "lag_features(df, column='value', lags=[1, 7, 30], group_by='id')",
    },
    {
        "name": "rolling_stats",
        "description": "Create rolling window statistics (mean, std, min, max)",
        "usage": "rolling_stats(df, column='value', windows=[7, 30], stats=['mean', 'std'])",
    },
    {
        "name": "datetime_features",
        "description": "Extract datetime components (hour, day, month, etc.)",
        "usage": "datetime_features(df, column='timestamp', features=['hour', 'day_of_week'])",
    },
    {
        "name": "train_lightgbm",
        "description": "Train LightGBM with early stopping",
        "usage": "train_lightgbm(X_train, y_train, X_val, y_val, params={...})",
    },
    {
        "name": "train_xgboost",
        "description": "Train XGBoost with early stopping",
        "usage": "train_xgboost(X_train, y_train, X_val, y_val, params={...})",
    },
    {
        "name": "cross_validate",
        "description": "K-fold cross-validation with OOF predictions",
        "usage": "cross_validate(X, y, model_fn='lightgbm', n_folds=5)",
    },
]


class CoderAgent(BaseAgent, ToolCallMixin):
    """
    Coder agent that generates feature engineering and training code.
    
    Responsibilities:
    - Implement feature engineering based on strategy
    - Write training pipelines
    - Generate submission files
    - Use available tools from the library
    """

    role = AgentRole.CODER

    def __init__(self, model: str | None = None):
        super().__init__(
            model=model,
            temperature=0.0,  # Deterministic code generation
            task_type="code_generation",
        )

    def get_system_prompt(self, context: dict[str, Any]) -> str:
        """Build system prompt with tools and context."""
        tools_description = self.format_tools_for_prompt(AVAILABLE_TOOLS)
        
        dataset_columns = context.get("columns", [])
        target_column = context.get("target_column", "target")
        metric = context.get("metric", "rmse")
        hypothesis = context.get("hypothesis", "No specific hypothesis")
        
        return f"""You are an expert ML engineer writing production-grade code for competitions.

## Available Tools
You have access to these pre-built tools from comet_swarm.tools:

{tools_description}

## Dataset Information
- **Columns**: {', '.join(dataset_columns[:20])}{'...' if len(dataset_columns) > 20 else ''}
- **Target**: {target_column}
- **Metric**: {metric}

## Current Hypothesis
{hypothesis}

## Code Requirements
1. Use **Polars** for data manipulation (import polars as pl)
2. Use the **available tools** from comet_swarm.tools
3. All code must be **executable** in a sandbox environment
4. Include proper **error handling**
5. Log important metrics to **Comet ML**
6. Keep code **concise and efficient**

## Output Format
Respond with executable Python code in a code block:

```python
# Your code here
```

Include a brief description of what the code does before the code block."""

    async def generate_feature_code(
        self,
        hypothesis: str,
        columns: list[str],
        target_column: str,
        dataset_sample: str | None = None,
    ) -> CodeArtifact:
        """
        Generate feature engineering code based on hypothesis.
        
        Args:
            hypothesis: Feature engineering hypothesis to implement
            columns: Available columns in the dataset
            target_column: Target column name
            dataset_sample: Optional sample of dataset for context
        
        Returns:
            CodeArtifact with generated code
        """
        context = {
            "columns": columns,
            "target_column": target_column,
            "hypothesis": hypothesis,
        }
        
        user_message = f"""Implement the following feature engineering hypothesis:

**Hypothesis**: {hypothesis}

Generate code that:
1. Loads data from 'data/train.parquet' using Polars
2. Creates the new features based on the hypothesis
3. Saves the enhanced dataset to 'data/train_features.parquet'
4. Prints summary of new features created

{f'**Dataset Sample**:{chr(10)}{dataset_sample}' if dataset_sample else ''}

Use the available tools from comet_swarm.tools where applicable."""

        response = await self.invoke(user_message, context)
        
        # Extract code from response
        import re
        code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        
        # Extract description (text before code block)
        description = response.split("```")[0].strip() if "```" in response else "Feature engineering code"
        
        return CodeArtifact(
            code=code,
            description=description[:200],
            dependencies=["polars", "comet_swarm"],
            expected_output="train_features.parquet with new columns",
        )

    async def generate_training_code(
        self,
        model_type: str,
        columns: list[str],
        target_column: str,
        metric: str,
        params: dict[str, Any] | None = None,
    ) -> CodeArtifact:
        """
        Generate model training code.
        
        Args:
            model_type: Model type (lightgbm, xgboost, catboost)
            columns: Feature columns
            target_column: Target column
            metric: Evaluation metric
            params: Optional hyperparameters
        
        Returns:
            CodeArtifact with training code
        """
        context = {
            "columns": columns,
            "target_column": target_column,
            "metric": metric,
        }
        
        user_message = f"""Generate training code for a {model_type} model.

**Configuration**:
- Model: {model_type}
- Target: {target_column}
- Metric: {metric}
- Features: {len(columns)} columns
{f'- Params: {params}' if params else ''}

The code should:
1. Load features from 'data/train_features.parquet'
2. Run 5-fold cross-validation using cross_validate()
3. Print CV scores and feature importance
4. Save OOF predictions to 'data/oof_predictions.parquet'
5. Train final model on all data
6. Save model to 'models/{model_type}_model.pkl'

Use comet_swarm.tools for training."""

        response = await self.invoke(user_message, context)
        
        import re
        code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        
        description = response.split("```")[0].strip() if "```" in response else f"{model_type} training code"
        
        return CodeArtifact(
            code=code,
            description=description[:200],
            dependencies=["polars", "comet_swarm", model_type],
            expected_output=f"models/{model_type}_model.pkl",
        )

    async def generate_submission_code(
        self,
        model_path: str,
        test_path: str,
        submission_format: str = "csv",
    ) -> CodeArtifact:
        """
        Generate submission code.
        
        Args:
            model_path: Path to trained model
            test_path: Path to test data
            submission_format: Output format (csv, parquet, python)
        
        Returns:
            CodeArtifact with submission generation code
        """
        context = {
            "model_path": model_path,
            "test_path": test_path,
            "submission_format": submission_format,
        }
        
        user_message = f"""Generate submission code.

**Configuration**:
- Model: {model_path}
- Test data: {test_path}
- Format: {submission_format}

The code should:
1. Load the trained model
2. Load test data and apply same feature engineering
3. Generate predictions
4. Create submission file in the required format
5. Validate submission format

{"For Wundernn (python format): Create solution.py with PredictionModel class" if submission_format == "python" else ""}"""

        response = await self.invoke(user_message, context)
        
        import re
        code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        
        return CodeArtifact(
            code=code_match.group(1) if code_match else response,
            description="Submission generation code",
            dependencies=["polars", "pickle"],
            expected_output=f"submission.{submission_format}",
        )
