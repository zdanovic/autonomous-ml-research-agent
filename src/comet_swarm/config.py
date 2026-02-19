"""Configuration management using Pydantic Settings."""

from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"  # Recommended: single key for all models


class ModelTier(str, Enum):
    """Model tiers for token optimization."""

    PREMIUM = "premium"  # GPT-4o, Claude 3.5 Sonnet
    STANDARD = "standard"  # GPT-4o-mini, Claude 3.5 Haiku
    FAST = "fast"  # GPT-4o-mini


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==========================================================================
    # API Keys (use OpenRouter OR direct provider keys)
    # ==========================================================================
    # OpenRouter (recommended - single key for all models)
    openrouter_api_key: SecretStr | None = Field(None, alias="OPENROUTER_API_KEY")
    
    # Direct provider keys (optional if using OpenRouter)
    openai_api_key: SecretStr | None = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(None, alias="ANTHROPIC_API_KEY")
    
    # Comet keys (Opik uses same key as Comet ML)
    comet_api_key: SecretStr = Field(..., alias="COMET_API_KEY")
    opik_api_key: SecretStr | None = Field(None, alias="OPIK_API_KEY")  # Falls back to comet_api_key
    
    # E2B Sandbox
    e2b_api_key: SecretStr = Field(..., alias="E2B_API_KEY")
    
    # Platform-specific (optional)
    kaggle_username: str | None = Field(None, alias="KAGGLE_USERNAME")
    kaggle_key: SecretStr | None = Field(None, alias="KAGGLE_KEY")

    @property
    def effective_opik_key(self) -> str:
        """Get Opik API key (falls back to Comet key)."""
        if self.opik_api_key:
            return self.opik_api_key.get_secret_value()
        return self.comet_api_key.get_secret_value()

    # ==========================================================================
    # LLM Configuration
    # ==========================================================================
    default_provider: LLMProvider = Field(default=LLMProvider.OPENROUTER)
    
    # Model mapping by tier (use OpenRouter format: provider/model)
    # For OpenRouter: openai/gpt-4o, anthropic/claude-3.5-sonnet, etc.
    # For direct: gpt-4o, claude-3-5-sonnet-20241022
    premium_model: str = Field(default="openai/gpt-4o")
    standard_model: str = Field(default="openai/gpt-4o-mini")
    fast_model: str = Field(default="openai/gpt-4o-mini")

    # ==========================================================================
    # Token Budget Optimization
    # ==========================================================================
    max_budget_usd: float = Field(default=10.0, description="Maximum token budget in USD")
    token_optimization_enabled: bool = Field(default=True)
    
    # Token costs per 1M tokens (input/output)
    token_costs: dict[str, dict[str, float]] = Field(
        default={
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
            "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        }
    )

    # Task type to tier mapping for smart model selection
    task_tier_mapping: dict[str, ModelTier] = Field(
        default={
            # Premium tasks (complex reasoning)
            "strategy_generation": ModelTier.PREMIUM,
            "complex_bug_fix": ModelTier.PREMIUM,
            "architecture_decision": ModelTier.PREMIUM,
            # Standard tasks
            "code_generation": ModelTier.STANDARD,
            "feature_engineering": ModelTier.STANDARD,
            "error_analysis": ModelTier.STANDARD,
            # Fast tasks (simple, high volume)
            "eda_summary": ModelTier.FAST,
            "code_formatting": ModelTier.FAST,
            "simple_extraction": ModelTier.FAST,
        }
    )

    # ==========================================================================
    # Comet Configuration
    # ==========================================================================
    comet_project_name: str = Field(default="comet-swarm")
    comet_workspace: str | None = Field(None)
    opik_project_name: str = Field(default="comet-swarm-agents")

    # ==========================================================================
    # Sandbox Configuration
    # ==========================================================================
    sandbox_timeout_seconds: int = Field(default=1800, description="30 minutes")
    sandbox_memory_mb: int = Field(default=8192, description="8 GB")
    sandbox_template: str = Field(default="Python3-DataScience")

    # ==========================================================================
    # Competition Settings
    # ==========================================================================
    max_iterations: int = Field(default=10, description="Max agent loop iterations")
    target_score_threshold: float | None = Field(
        None, description="Stop if score reaches this threshold"
    )

    def get_model_for_task(self, task_type: str) -> str:
        """Get the appropriate model based on task type and budget optimization."""
        if not self.token_optimization_enabled:
            return self.premium_model

        tier = self.task_tier_mapping.get(task_type, ModelTier.STANDARD)
        
        match tier:
            case ModelTier.PREMIUM:
                return self.premium_model
            case ModelTier.STANDARD:
                return self.standard_model
            case ModelTier.FAST:
                return self.fast_model

    def get_token_cost(self, model: str) -> dict[str, float]:
        """Get token cost for a model."""
        return self.token_costs.get(model, {"input": 5.0, "output": 15.0})


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
