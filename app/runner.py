"""
Agent Runner - Background service for running the agent with real-time updates.
"""

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

# Add src directory to path for Streamlit environment
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl

from comet_swarm.agents import StrategistAgent
from comet_swarm.integrations import ExperimentTracker, get_tracer
from comet_swarm.platforms import KaggleAdapter, SolafuneAdapter, WundernnAdapter
from comet_swarm.tools import (
    cross_validate,
    lag_features,
    profile_dataset,
)


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class Phase(str, Enum):
    EDA = "EDA"
    STRATEGY = "Strategy"
    FEATURES = "Features"
    TRAINING = "Training"
    EVALUATION = "Evaluation"
    SUBMIT = "Submit"


@dataclass
class RunConfig:
    """Configuration for an agent run."""
    platform: str
    mode: str
    data_path: str | None = None
    max_iterations: int = 10
    token_budget: float = 10.0
    n_folds: int = 5
    sample_size: int | None = None


@dataclass
class Checkpoint:
    """A human-in-the-loop checkpoint."""
    id: str
    title: str
    description: str
    options: list[str]
    response: str | None = None


class AgentRunner:
    """
    Runs the agent workflow with real-time updates.
    
    This class manages the agent lifecycle and provides callbacks
    for the Streamlit UI to receive updates.
    """
    
    PLATFORM_ADAPTERS = {
        "wundernn": WundernnAdapter,
        "solafune": SolafuneAdapter,
        "kaggle": KaggleAdapter,
    }
    
    MODE_CONFIGS = {
        "debug": {"max_iterations": 1, "n_folds": 2, "sample_size": 1000},
        "test": {"max_iterations": 3, "n_folds": 3, "sample_size": 10000},
        "full": {"max_iterations": 10, "n_folds": 5, "sample_size": None},
    }
    
    def __init__(
        self,
        on_log: Callable[[str, str], None] | None = None,
        on_phase_change: Callable[[Phase], None] | None = None,
        on_checkpoint: Callable[[Checkpoint], None] | None = None,
        on_experiment: Callable[[dict], None] | None = None,
        on_status_change: Callable[[AgentStatus], None] | None = None,
    ):
        """
        Initialize agent runner with callbacks.
        
        Args:
            on_log: Callback for log messages (message, level)
            on_phase_change: Callback when phase changes
            on_checkpoint: Callback when checkpoint is reached
            on_experiment: Callback when experiment is logged
            on_status_change: Callback when status changes
        """
        self.on_log = on_log or (lambda msg, lvl: print(f"[{lvl}] {msg}"))
        self.on_phase_change = on_phase_change or (lambda p: None)
        self.on_checkpoint = on_checkpoint or (lambda c: None)
        self.on_experiment = on_experiment or (lambda e: None)
        self.on_status_change = on_status_change or (lambda s: None)
        
        self.status = AgentStatus.IDLE
        self.current_phase: Phase | None = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._stop_requested = False
        
        # Checkpoint handling
        self._pending_checkpoint: Checkpoint | None = None
        self._checkpoint_event = asyncio.Event()
    
    def log(self, message: str, level: str = "info"):
        """Log a message."""
        self.on_log(message, level)
    
    def set_phase(self, phase: Phase):
        """Set current phase."""
        self.current_phase = phase
        self.on_phase_change(phase)
        self.log(f"Phase: {phase.value}", "info")
    
    def set_status(self, status: AgentStatus):
        """Set agent status."""
        self.status = status
        self.on_status_change(status)
    
    def pause(self):
        """Pause the agent."""
        self._pause_event.clear()
        self.set_status(AgentStatus.PAUSED)
        self.log("Agent paused", "warning")
    
    def resume(self):
        """Resume the agent."""
        self._pause_event.set()
        self.set_status(AgentStatus.RUNNING)
        self.log("Agent resumed", "info")
    
    def stop(self):
        """Stop the agent."""
        self._stop_requested = True
        self._pause_event.set()  # Unblock if paused
        self.log("Stop requested", "warning")
    
    async def _check_pause(self):
        """Check if paused and wait if so."""
        await self._pause_event.wait()
        if self._stop_requested:
            raise asyncio.CancelledError("Agent stopped by user")
    
    async def request_checkpoint(
        self,
        title: str,
        description: str,
        options: list[str],
    ) -> str:
        """
        Request human input at a checkpoint.
        
        Args:
            title: Checkpoint title
            description: Description for human
            options: List of options to choose from
            
        Returns:
            Selected option
        """
        checkpoint = Checkpoint(
            id=f"cp_{datetime.now().timestamp()}",
            title=title,
            description=description,
            options=options,
        )
        
        self._pending_checkpoint = checkpoint
        self._checkpoint_event.clear()
        
        self.on_checkpoint(checkpoint)
        self.log(f"Checkpoint: {title}", "warning")
        
        # Wait for response
        await self._checkpoint_event.wait()
        
        response = self._pending_checkpoint.response
        self._pending_checkpoint = None
        
        return response or options[0]
    
    def respond_to_checkpoint(self, checkpoint_id: str, response: str):
        """Respond to a pending checkpoint."""
        if self._pending_checkpoint and self._pending_checkpoint.id == checkpoint_id:
            self._pending_checkpoint.response = response
            self._checkpoint_event.set()
            self.log(f"Checkpoint resolved: {response}", "success")
    
    async def run(self, config: RunConfig) -> dict:
        """
        Run the agent with the given configuration.
        
        Args:
            config: Run configuration
            
        Returns:
            Results dictionary
        """
        self._stop_requested = False
        self.set_status(AgentStatus.RUNNING)
        
        try:
            # Apply mode config
            mode_cfg = self.MODE_CONFIGS.get(config.mode, self.MODE_CONFIGS["debug"])
            config.max_iterations = mode_cfg["max_iterations"]
            config.n_folds = mode_cfg["n_folds"]
            config.sample_size = mode_cfg["sample_size"]
            
            self.log(f"Starting run: {config.platform} / {config.mode}", "success")
            
            # Initialize
            tracer = get_tracer()
            tracer.start_trace(f"{config.platform}_dashboard_run")
            
            tracker = ExperimentTracker(
                project_name="comet-swarm",
                competition_name=f"{config.platform}_competition",
            )
            
            # Get adapter
            adapter_cls = self.PLATFORM_ADAPTERS.get(config.platform)
            if not adapter_cls:
                raise ValueError(f"Unknown platform: {config.platform}")
            
            adapter = adapter_cls()
            context = adapter.get_competition_context()
            target_col = context.target_column or "target"
            
            # Phase 1: EDA
            await self._check_pause()
            self.set_phase(Phase.EDA)
            
            # Load data
            df = await self._load_data(config, adapter)
            if df is None:
                raise ValueError("Failed to load data")
            
            self.log(f"Loaded {len(df):,} rows, {len(df.columns)} columns", "success")
            
            # Profile
            try:
                dataset_info = profile_dataset(df, name=config.platform, target_column=target_col)
                self.log(f"Memory: {dataset_info.memory_mb:.2f} MB", "info")
            except Exception as e:
                self.log(f"EDA warning: {e}", "warning")
                dataset_info = None
            
            # Phase 2: Strategy
            await self._check_pause()
            self.set_phase(Phase.STRATEGY)
            
            strategist = StrategistAgent()
            
            try:
                strategy = await strategist.generate_strategy(
                    competition_context=context.model_dump(),
                    dataset_info=f"{len(df)} rows, {len(df.columns)} cols",
                    eda_insights="See EDA phase",
                )
                self.log(f"Strategy: {len(strategy.hypotheses)} hypotheses", "success")
            except Exception as e:
                self.log(f"Strategy error: {e}", "error")
                strategy = None
            
            # Human checkpoint for strategy approval
            if strategy:
                response = await self.request_checkpoint(
                    title="Approve Strategy?",
                    description=f"Priority: {strategy.priority[:100]}...\nHypotheses: {len(strategy.hypotheses)}",
                    options=["Approve", "Skip to Training", "Stop"],
                )
                
                if response == "Stop":
                    self.set_status(AgentStatus.IDLE)
                    return {"status": "stopped"}
            
            # Phase 3: Features
            await self._check_pause()
            self.set_phase(Phase.FEATURES)
            
            numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric() and c != target_col]
            
            if numeric_cols:
                try:
                    feature_col = numeric_cols[0]
                    df = lag_features(df, column=feature_col, lags=[1, 3])
                    new_features = [c for c in df.columns if "lag" in c]
                    self.log(f"Created {len(new_features)} features", "success")
                except Exception as e:
                    self.log(f"Feature error: {e}", "warning")
            
            # Phase 4: Training
            await self._check_pause()
            self.set_phase(Phase.TRAINING)
            
            feature_cols = [c for c in df.columns if c != target_col and df[c].dtype.is_numeric()]
            
            if not feature_cols or target_col not in df.columns:
                self.log("No features or target for training", "error")
                self.set_status(AgentStatus.ERROR)
                return {"status": "error", "message": "No features or target"}
            
            df_train = df.select(feature_cols + [target_col]).drop_nulls()
            X = df_train.select(feature_cols).to_numpy()
            y = df_train.select(target_col).to_numpy().flatten()
            
            self.log(f"Training: {len(X):,} samples, {len(feature_cols)} features", "info")
            
            tracker.start_experiment(
                name=f"{config.platform}_{config.mode}_dashboard",
                tags=[config.platform, config.mode, "dashboard"],
            )
            
            try:
                cv_result = cross_validate(
                    X, y,
                    model_fn="lightgbm",
                    params={"num_leaves": 31, "learning_rate": 0.05, "verbose": -1},
                    n_folds=config.n_folds,
                    log_to_comet=True,
                )
                
                self.log(f"CV: {cv_result['cv_mean']:.4f} ± {cv_result['cv_std']:.4f}", "success")
                
                experiment_data = {
                    "name": f"lgbm_{config.platform}_{config.mode}",
                    "platform": config.platform,
                    "cv_mean": cv_result["cv_mean"],
                    "cv_std": cv_result["cv_std"],
                    "features": len(feature_cols),
                    "created": datetime.now().isoformat(),
                }
                self.on_experiment(experiment_data)
                
            except Exception as e:
                self.log(f"Training error: {e}", "error")
                cv_result = None
            
            experiment_result = tracker.end_experiment()
            if experiment_result:
                self.log(f"Experiment: {experiment_result.experiment_url}", "info")
            
            # Phase 5: Evaluation
            await self._check_pause()
            self.set_phase(Phase.EVALUATION)
            
            # Ask about submission
            submit_response = await self.request_checkpoint(
                title="Submit to Leaderboard?",
                description=f"CV Score: {cv_result['cv_mean']:.4f}" if cv_result else "No CV result",
                options=["Submit", "Skip", "Another Iteration"],
            )
            
            # Phase 6: Submit (if approved)
            if submit_response == "Submit":
                self.set_phase(Phase.SUBMIT)
                self.log("Submission would happen here", "info")
            elif submit_response == "Another Iteration":
                self.log("Would start another iteration", "info")
            
            # Done
            budget = tracer.get_budget_status()
            self.log(f"Token budget: ${budget.total_spent_usd:.4f} used", "info")
            
            self.set_status(AgentStatus.COMPLETED)
            return {
                "status": "completed",
                "cv_score": cv_result["cv_mean"] if cv_result else None,
                "budget_used": budget.total_spent_usd,
            }
            
        except asyncio.CancelledError:
            self.set_status(AgentStatus.IDLE)
            return {"status": "stopped"}
        except Exception as e:
            self.log(f"Error: {e}", "error")
            self.set_status(AgentStatus.ERROR)
            return {"status": "error", "message": str(e)}
    
    async def _load_data(self, config: RunConfig, adapter) -> pl.DataFrame | None:
        """Load data from path or adapter."""
        
        if config.data_path and Path(config.data_path).exists():
            path = Path(config.data_path)
            self.log(f"Loading: {path.name}", "info")
            
            if path.suffix == ".parquet":
                df = pl.read_parquet(path)
            else:
                df = pl.read_csv(path)
        else:
            # Try to download or use synthetic
            self.log("Using synthetic data for demo", "warning")
            
            # Create synthetic data
            import numpy as np
            n = config.sample_size or 10000
            
            df = pl.DataFrame({
                f"p{i}": np.random.randn(n) for i in range(10)
            })
            df = df.with_columns([
                (pl.col("p0") * 0.5 + pl.col("p1") * 0.3 + pl.Series(np.random.randn(n) * 0.2)).alias("target")
            ])
        
        # Sample if needed
        if config.sample_size and len(df) > config.sample_size:
            df = df.sample(n=config.sample_size, seed=42)
            self.log(f"Sampled to {config.sample_size:,} rows", "info")
        
        return df


# Global runner instance for Streamlit
_runner: AgentRunner | None = None


def get_runner() -> AgentRunner:
    """Get or create the global runner instance."""
    global _runner
    if _runner is None:
        _runner = AgentRunner()
    return _runner
