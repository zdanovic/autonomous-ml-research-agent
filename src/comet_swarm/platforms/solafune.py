"""Solafune platform adapter."""

from pathlib import Path
from typing import Any

import httpx

from comet_swarm.platforms.base import BasePlatformAdapter
from comet_swarm.schemas import CompetitionContext, MetricType, Platform, SubmissionResult


class SolafuneAdapter(BasePlatformAdapter):
    """
    Adapter for Solafune competitions.
    
    Key characteristics:
    - Data: Multimodal (tabular + satellite imagery)
    - Large datasets (17GB+)
    - Submission: CSV format
    - Metric: RMSLE
    """

    platform_name = "solafune"
    requires_auth = True

    def __init__(
        self,
        api_key: str | None = None,
        competition_id: str = "1918ccd7-eb06-4cfc-822f-a9823c63b2c1",
    ):
        """
        Initialize Solafune adapter.
        
        Args:
            api_key: API key for Solafune
            competition_id: UUID of the competition
        """
        self.api_key = api_key
        self.competition_id = competition_id
        self.base_url = "https://solafune.com/api/v1"
        self._client: httpx.AsyncClient | None = None

    async def authenticate(self) -> bool:
        """Authenticate with Solafune platform."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=600.0,  # Long timeout for large files
        )
        
        # Verify auth
        try:
            response = await self._client.get("/user/me")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def download_data(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Download competition data.
        
        Note: Solafune datasets are large (17GB+). This method
        handles chunked downloads for reliability.
        
        Args:
            output_dir: Directory to save data files
        
        Returns:
            Paths to downloaded data files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self._client:
            await self.authenticate()

        result = {}
        data_files = ["train_dataset.zip", "evaluation_dataset.zip"]

        for filename in data_files:
            file_path = output_dir / filename
            
            try:
                # Stream download for large files
                async with self._client.stream(
                    "GET",
                    f"/competitions/{self.competition_id}/data/{filename}",
                ) as response:
                    response.raise_for_status()
                    
                    with open(file_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                
                result[filename.replace(".zip", "")] = file_path
                
            except httpx.HTTPError as e:
                print(f"Failed to download {filename}: {e}")

        return result

    async def submit(
        self,
        prediction_path: str | Path,
        message: str = "",
    ) -> SubmissionResult:
        """
        Submit predictions to Solafune.
        
        Args:
            prediction_path: Path to CSV submission file
            message: Optional submission message
        
        Returns:
            SubmissionResult
        """
        prediction_path = Path(prediction_path)

        if not self._client:
            await self.authenticate()

        try:
            with open(prediction_path, "rb") as f:
                response = await self._client.post(
                    f"/competitions/{self.competition_id}/submissions",
                    files={"file": (prediction_path.name, f, "text/csv")},
                    data={"description": message},
                )
            response.raise_for_status()
            data = response.json()

            return SubmissionResult(
                submission_id=data.get("id"),
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
            response = await self._client.get(
                f"/competitions/{self.competition_id}/leaderboard",
                params={"limit": top_n},
            )
            response.raise_for_status()
            return response.json().get("entries", [])
        except httpx.HTTPError:
            return []

    def get_competition_context(self) -> CompetitionContext:
        """Get Solafune competition context."""
        return CompetitionContext(
            name="Construction Cost Prediction (Solafune)",
            platform=Platform.SOLAFUNE,
            description=(
                "Estimate construction costs per square meter by integrating "
                "macroeconomic indicators with satellite-derived remote sensing data"
            ),
            metric=MetricType.RMSLE,
            metric_direction="minimize",
            target_column="cost_per_sqm",
            submission_format="csv",
        )

    def get_eda_strategy(self) -> dict[str, list[str]]:
        """Get EDA strategy for multimodal data."""
        return {
            "tabular_analysis": [
                "profile_dataset",
                "correlation_analysis",
                "distribution_analysis",
                "economic_indicator_trends",
            ],
            "image_analysis": [
                "sample_visualization",
                "building_density_stats",
                "land_use_distribution",
                "spatial_clustering",
            ],
            "multimodal_analysis": [
                "feature_correlation_heatmap",
                "geographic_distribution",
            ],
        }

    def get_model_recommendations(self) -> list[str]:
        """Get model recommendations for multimodal data."""
        return [
            "LightGBM for tabular features (baseline)",
            "EfficientNet-B0 for satellite image embeddings",
            "ResNet-18 pretrained on satellite data",
            "Multimodal fusion with late fusion strategy",
            "Stacking ensemble for final predictions",
        ]


class SolafuneDataLoader:
    """Helper class for loading Solafune multimodal data."""

    def __init__(self, data_dir: str | Path):
        """Initialize data loader."""
        self.data_dir = Path(data_dir)

    def load_tabular(self, split: str = "train") -> "pl.DataFrame":
        """Load tabular data."""
        import polars as pl
        return pl.read_csv(self.data_dir / f"{split}.csv")

    def load_image_paths(self, split: str = "train") -> list[Path]:
        """Get paths to satellite images."""
        image_dir = self.data_dir / "images" / split
        if image_dir.exists():
            return list(image_dir.glob("*.tif"))
        return []

    def get_image_for_sample(self, sample_id: str, split: str = "train") -> Path | None:
        """Get image path for a specific sample."""
        image_path = self.data_dir / "images" / split / f"{sample_id}.tif"
        return image_path if image_path.exists() else None
