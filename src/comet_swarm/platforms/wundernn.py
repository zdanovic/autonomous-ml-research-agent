"""Wundernn Predictorium platform adapter."""

import io
import zipfile
from pathlib import Path
from typing import Any

import httpx

from comet_swarm.platforms.base import BasePlatformAdapter
from comet_swarm.schemas import CompetitionContext, MetricType, Platform, SubmissionResult


class WundernnAdapter(BasePlatformAdapter):
    """
    Adapter for Wundernn Predictorium (LOB Prediction) competition.
    
    Key characteristics:
    - Data format: Parquet files
    - Submission: Python code (zip with solution.py)
    - Metric: Weighted Pearson Correlation
    """

    platform_name = "wundernn"
    requires_auth = True

    # Solution template for submission
    SOLUTION_TEMPLATE = '''"""Wundernn Predictorium Solution."""

import numpy as np
import pickle
from pathlib import Path


class PredictionModel:
    """Model class for Wundernn submission."""

    def __init__(self):
        """Load the trained model."""
        model_path = Path(__file__).parent / "model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, data_point: dict) -> np.ndarray | None:
        """
        Generate predictions for a single data point.

        Args:
            data_point: Dictionary with keys 'seq_ix', 'step_in_seq', 
                       'need_prediction', and 'p0' through 'p11'

        Returns:
            np.ndarray of shape (2,) with [t0_pred, t1_pred] or None
        """
        if not data_point.get("need_prediction", True):
            return None

        # Extract features
        features = np.array([
            data_point[f"p{i}"] for i in range(12)
        ]).reshape(1, -1)

        # Generate predictions
        prediction = self.model.predict(features)

        return prediction[0] if prediction.ndim > 1 else prediction
'''

    def __init__(
        self,
        api_key: str | None = None,
        competition_id: str = "lob_predictorium",
    ):
        """
        Initialize Wundernn adapter.
        
        Args:
            api_key: Optional API key (from env if not provided)
            competition_id: Competition identifier
        """
        self.api_key = api_key
        self.competition_id = competition_id
        self.base_url = "https://wundernn.io/predictorium/api"
        self._client: httpx.AsyncClient | None = None

    async def authenticate(self) -> bool:
        """Authenticate with Wundernn platform."""
        # Wundernn may not require explicit auth for public data
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=60.0,
        )
        return True

    async def download_data(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Download competition data.
        
        Args:
            output_dir: Directory to save data files
        
        Returns:
            Paths to train and test parquet files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Note: Actual URLs would come from Wundernn API
        # This is a placeholder implementation
        data_urls = {
            "train": f"{self.base_url}/data/train.parquet",
            "test": f"{self.base_url}/data/test.parquet",
        }

        result = {}
        async with httpx.AsyncClient(timeout=300.0) as client:
            for name, url in data_urls.items():
                try:
                    response = await client.get(url)
                    response.raise_for_status()

                    file_path = output_dir / f"{name}.parquet"
                    file_path.write_bytes(response.content)
                    result[name] = file_path
                except httpx.HTTPError as e:
                    print(f"Failed to download {name}: {e}")

        return result

    async def submit(
        self,
        prediction_path: str | Path,
        message: str = "",
    ) -> SubmissionResult:
        """
        Submit solution to Wundernn.
        
        For Wundernn, prediction_path should be a directory containing:
        - solution.py (or we generate from template)
        - model.pkl (trained model)
        
        Args:
            prediction_path: Path to submission directory
            message: Optional submission message
        
        Returns:
            SubmissionResult
        """
        prediction_path = Path(prediction_path)

        # Create submission zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            if prediction_path.is_dir():
                # Package directory contents
                for file in prediction_path.iterdir():
                    zf.write(file, file.name)
            else:
                # Assume it's a model file, use template
                zf.writestr("solution.py", self.SOLUTION_TEMPLATE)
                zf.write(prediction_path, "model.pkl")

        zip_buffer.seek(0)

        # Submit to API
        if not self._client:
            await self.authenticate()

        try:
            response = await self._client.post(
                "/submit",
                files={"submission": ("submission.zip", zip_buffer, "application/zip")},
                data={"message": message},
            )
            response.raise_for_status()
            data = response.json()

            return SubmissionResult(
                submission_id=data.get("submission_id"),
                status=data.get("status", "pending"),
                public_score=data.get("score"),
                message=data.get("message"),
            )
        except httpx.HTTPError as e:
            return SubmissionResult(
                status="failed",
                message=f"Submission failed: {e!s}",
            )

    async def get_leaderboard(self, top_n: int = 50) -> list[dict[str, Any]]:
        """Get current leaderboard."""
        if not self._client:
            await self.authenticate()

        try:
            response = await self._client.get(f"/leaderboard?limit={top_n}")
            response.raise_for_status()
            return response.json().get("entries", [])
        except httpx.HTTPError:
            return []

    def get_competition_context(self) -> CompetitionContext:
        """Get Wundernn competition context."""
        return CompetitionContext(
            name="Wunder Challenge: LOB Predictorium",
            platform=Platform.WUNDERNN,
            description="Predict Limit Order Book (LOB) price changes for high-frequency trading",
            metric=MetricType.PEARSON,
            metric_direction="maximize",
            target_column="t0,t1",  # Dual target
            submission_format="python",
        )

    def get_eda_strategy(self) -> dict[str, list[str]]:
        """Get EDA strategy for LOB data."""
        return {
            "tabular_analysis": [
                "profile_dataset",
                "correlation_analysis",
                "temporal_patterns",
            ],
            "sequence_analysis": [
                "autocorrelation",
                "lag_correlations",
                "stationarity_tests",
            ],
        }

    def get_model_recommendations(self) -> list[str]:
        """Get model recommendations for LOB prediction."""
        return [
            "LightGBM with lag features",
            "1D-CNN for sequence patterns",
            "LSTM/GRU for temporal dependencies",
            "Transformer encoder for attention",
        ]

    def create_submission_package(
        self,
        model_path: str | Path,
        output_path: str | Path,
        custom_solution_code: str | None = None,
    ) -> Path:
        """
        Create submission zip package.
        
        Args:
            model_path: Path to trained model (pkl file)
            output_path: Path for output zip file
            custom_solution_code: Optional custom solution.py content
        
        Returns:
            Path to created zip file
        """
        model_path = Path(model_path)
        output_path = Path(output_path)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add solution.py
            solution_code = custom_solution_code or self.SOLUTION_TEMPLATE
            zf.writestr("solution.py", solution_code)

            # Add model
            zf.write(model_path, "model.pkl")

        return output_path
