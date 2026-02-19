"""Kaggle platform adapter."""

import os
from pathlib import Path
from typing import Any

from comet_swarm.config import get_settings
from comet_swarm.platforms.base import BasePlatformAdapter
from comet_swarm.schemas import CompetitionContext, MetricType, Platform, SubmissionResult


class KaggleAdapter(BasePlatformAdapter):
    """
    Adapter for Kaggle competitions.
    
    Uses the official Kaggle API with rate limiting
    and caching.
    """

    platform_name = "kaggle"
    requires_auth = True

    def __init__(
        self,
        competition_slug: str = "urban-flood-modelling",
        username: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize Kaggle adapter.
        
        Args:
            competition_slug: Kaggle competition slug
            username: Kaggle username (from env if not provided)
            api_key: Kaggle API key (from env if not provided)
        """
        self.competition_slug = competition_slug
        
        settings = get_settings()
        self.username = username or settings.kaggle_username
        self.api_key = api_key or (
            settings.kaggle_key.get_secret_value() if settings.kaggle_key else None
        )
        
        self._api = None
        self._last_request_time = 0.0
        self._min_request_interval = 2.0  # Rate limiting

    async def authenticate(self) -> bool:
        """Authenticate with Kaggle API."""
        try:
            # Set up Kaggle credentials
            if self.username and self.api_key:
                os.environ["KAGGLE_USERNAME"] = self.username
                os.environ["KAGGLE_KEY"] = self.api_key
            
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            self._api = KaggleApi()
            self._api.authenticate()
            return True
        except Exception as e:
            print(f"Kaggle authentication failed: {e}")
            return False

    def _rate_limit(self) -> None:
        """Apply rate limiting to API calls."""
        import time
        
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    async def download_data(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Download competition data.
        
        Args:
            output_dir: Directory to save data files
        
        Returns:
            Dictionary of file type to path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self._api:
            await self.authenticate()

        self._rate_limit()
        
        try:
            # Download all competition files
            self._api.competition_download_files(
                self.competition_slug,
                path=str(output_dir),
                quiet=False,
            )
            
            # Find downloaded files
            result = {}
            for file in output_dir.iterdir():
                if "train" in file.name.lower():
                    result["train"] = file
                elif "test" in file.name.lower():
                    result["test"] = file
                elif "sample" in file.name.lower():
                    result["sample_submission"] = file
            
            return result
            
        except Exception as e:
            print(f"Data download failed: {e}")
            return {}

    async def submit(
        self,
        prediction_path: str | Path,
        message: str = "",
    ) -> SubmissionResult:
        """
        Submit predictions to Kaggle.
        
        Args:
            prediction_path: Path to submission CSV
            message: Submission description
        
        Returns:
            SubmissionResult
        """
        prediction_path = Path(prediction_path)

        if not self._api:
            await self.authenticate()

        self._rate_limit()
        
        try:
            self._api.competition_submit(
                file_name=str(prediction_path),
                message=message or "Comet-Swarm submission",
                competition=self.competition_slug,
                quiet=False,
            )
            
            # Get latest submission status
            self._rate_limit()
            submissions = self._api.competition_submissions(self.competition_slug)
            
            if submissions:
                latest = submissions[0]
                return SubmissionResult(
                    submission_id=str(latest.ref),
                    status=latest.status,
                    public_score=float(latest.publicScore) if latest.publicScore else None,
                    message=latest.description,
                )
            
            return SubmissionResult(
                status="pending",
                message="Submission uploaded",
            )
            
        except Exception as e:
            return SubmissionResult(
                status="failed",
                message=f"Submission failed: {e!s}",
            )

    async def get_leaderboard(self, top_n: int = 50) -> list[dict[str, Any]]:
        """Get current leaderboard."""
        if not self._api:
            await self.authenticate()

        self._rate_limit()
        
        try:
            lb = self._api.competition_leaderboard_view(self.competition_slug)
            
            entries = []
            for i, row in enumerate(lb[:top_n]):
                entries.append({
                    "rank": i + 1,
                    "team": row.teamName,
                    "score": row.score,
                    "entries": row.submissionCount,
                })
            
            return entries
            
        except Exception:
            return []

    def get_competition_context(self) -> CompetitionContext:
        """Get Kaggle Urban Flood competition context."""
        return CompetitionContext(
            name="Urban Flood Modelling (UrbanFloodBench)",
            platform=Platform.KAGGLE,
            description=(
                "Predict water levels autoregressively in urban drainage systems "
                "(1D pipes and 2D surface flow) during rainfall events"
            ),
            metric=MetricType.R2,
            metric_direction="maximize",
            target_column="water_level",
            submission_format="csv",
            api_endpoint=f"https://www.kaggle.com/competitions/{self.competition_slug}",
        )

    def get_eda_strategy(self) -> dict[str, list[str]]:
        """Get EDA strategy for flood modelling data."""
        return {
            "tabular_analysis": [
                "profile_dataset",
                "temporal_patterns",
                "correlation_analysis",
            ],
            "time_series_analysis": [
                "autocorrelation_plots",
                "seasonality_detection",
                "stationarity_tests",
            ],
            "graph_analysis": [
                "network_topology",
                "node_connectivity",
                "flow_patterns",
            ],
        }

    def get_model_recommendations(self) -> list[str]:
        """Get model recommendations for autoregressive flood prediction."""
        return [
            "Temporal Fusion Transformer (TFT)",
            "Graph Neural Network for network topology",
            "LSTM with attention for autoregressive",
            "Physics-informed neural network",
            "N-BEATS for time series baseline",
        ]

    async def get_my_submissions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get user's submission history."""
        if not self._api:
            await self.authenticate()

        self._rate_limit()
        
        try:
            submissions = self._api.competition_submissions(self.competition_slug)
            
            result = []
            for sub in submissions[:limit]:
                result.append({
                    "id": sub.ref,
                    "date": str(sub.date),
                    "status": sub.status,
                    "public_score": sub.publicScore,
                    "private_score": sub.privateScore,
                    "message": sub.description,
                })
            
            return result
            
        except Exception:
            return []
