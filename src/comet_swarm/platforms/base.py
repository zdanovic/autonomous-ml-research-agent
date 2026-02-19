"""Base platform adapter interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from comet_swarm.schemas import CompetitionContext, SubmissionResult


class BasePlatformAdapter(ABC):
    """Abstract base class for competition platform adapters."""

    platform_name: str
    requires_auth: bool = True

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the platform."""
        pass

    @abstractmethod
    async def download_data(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Download competition data.
        
        Args:
            output_dir: Directory to save data files
        
        Returns:
            Dictionary mapping file type to path (e.g., {"train": Path(...), "test": Path(...)})
        """
        pass

    @abstractmethod
    async def submit(
        self,
        prediction_path: str | Path,
        message: str = "",
    ) -> SubmissionResult:
        """
        Submit predictions to the competition.
        
        Args:
            prediction_path: Path to prediction file
            message: Optional submission message
        
        Returns:
            SubmissionResult with status and scores
        """
        pass

    @abstractmethod
    async def get_leaderboard(self, top_n: int = 50) -> list[dict[str, Any]]:
        """
        Get current leaderboard.
        
        Args:
            top_n: Number of top entries to return
        
        Returns:
            List of leaderboard entries
        """
        pass

    @abstractmethod
    def get_competition_context(self) -> CompetitionContext:
        """Get competition metadata as CompetitionContext."""
        pass

    def get_eda_strategy(self) -> dict[str, list[str]]:
        """Get recommended EDA strategy for this platform/competition."""
        return {
            "tabular_analysis": [
                "profile_dataset",
                "correlation_analysis",
                "distribution_analysis",
            ],
            "visualization": [
                "target_distribution",
                "feature_importance_plot",
            ],
        }

    def get_model_recommendations(self) -> list[str]:
        """Get model recommendations for this competition type."""
        return [
            "lightgbm",
            "xgboost",
            "catboost",
        ]
