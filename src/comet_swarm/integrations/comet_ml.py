"""Comet ML integration for experiment tracking."""

from pathlib import Path
from typing import Any

from comet_ml import Experiment

from comet_swarm.config import get_settings
from comet_swarm.schemas import ExperimentResult


class ExperimentTracker:
    """Wrapper for Comet ML experiment tracking."""

    def __init__(
        self,
        project_name: str | None = None,
        workspace: str | None = None,
        competition_name: str | None = None,
    ):
        settings = get_settings()
        
        self.project_name = project_name or settings.comet_project_name
        self.workspace = workspace or settings.comet_workspace
        self.competition_name = competition_name
        self._api_key = settings.comet_api_key.get_secret_value()
        
        # Current experiment
        self._experiment: Experiment | None = None
        self._experiment_count = 0

    def start_experiment(
        self,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> Experiment:
        """Start a new experiment."""
        self._experiment_count += 1
        
        experiment = Experiment(
            api_key=self._api_key,
            project_name=self.project_name,
            workspace=self.workspace,
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
        )
        
        # Set name
        if name:
            experiment.set_name(name)
        else:
            experiment.set_name(f"experiment_{self._experiment_count}")
        
        # Add tags
        if tags:
            experiment.add_tags(tags)
        
        if self.competition_name:
            experiment.add_tag(self.competition_name)
        
        self._experiment = experiment
        return experiment

    def log_parameters(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self._experiment:
            self._experiment.log_parameters(params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Log metrics."""
        if self._experiment:
            self._experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Log a single metric."""
        if self._experiment:
            self._experiment.log_metric(name, value, step=step, epoch=epoch)

    def log_feature_importance(
        self,
        feature_importance: dict[str, float],
        top_n: int = 20,
    ) -> None:
        """Log feature importance as a table."""
        if not self._experiment:
            return
        
        # Sort and take top N
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        
        # Log as table
        self._experiment.log_table(
            filename="feature_importance.csv",
            tabular_data=[
                {"feature": name, "importance": importance}
                for name, importance in sorted_features
            ],
        )

    def log_training_curve(
        self,
        train_losses: list[float],
        val_losses: list[float] | None = None,
        metric_name: str = "loss",
    ) -> None:
        """Log training and validation curves."""
        if not self._experiment:
            return
        
        for epoch, train_loss in enumerate(train_losses):
            self._experiment.log_metric(f"train_{metric_name}", train_loss, epoch=epoch)
        
        if val_losses:
            for epoch, val_loss in enumerate(val_losses):
                self._experiment.log_metric(f"val_{metric_name}", val_loss, epoch=epoch)

    def log_confusion_matrix(
        self,
        y_true: list[Any],
        y_pred: list[Any],
        labels: list[str] | None = None,
    ) -> None:
        """Log confusion matrix for classification tasks."""
        if self._experiment:
            self._experiment.log_confusion_matrix(
                y_true=y_true,
                y_predicted=y_pred,
                labels=labels,
            )

    def log_model(
        self,
        model_path: str | Path,
        model_name: str | None = None,
    ) -> None:
        """Log model artifact."""
        if self._experiment:
            self._experiment.log_model(
                name=model_name or "model",
                file_or_folder=str(model_path),
            )

    def log_code(self, code: str, filename: str = "generated_code.py") -> None:
        """Log generated code as an asset."""
        if self._experiment:
            self._experiment.log_asset_data(
                data=code,
                name=filename,
            )

    def log_dataset_info(
        self,
        name: str,
        num_rows: int,
        num_columns: int,
        column_names: list[str],
    ) -> None:
        """Log dataset information."""
        if self._experiment:
            self._experiment.log_dataset_info(
                name=name,
                path=name,
            )
            self._experiment.log_other("num_rows", num_rows)
            self._experiment.log_other("num_columns", num_columns)
            self._experiment.log_other("columns", ", ".join(column_names[:20]))

    def end_experiment(self) -> ExperimentResult | None:
        """End current experiment and return result."""
        if not self._experiment:
            return None
        
        experiment_url = self._experiment.url
        experiment_id = self._experiment.id
        
        # Get logged data
        metrics = {}
        parameters = {}
        
        try:
            # These may fail if not set
            metrics = {k: v for k, v in self._experiment.get_metrics_summary().items()}
        except Exception:
            pass
        
        try:
            parameters = dict(self._experiment.get_parameters_summary())
        except Exception:
            pass
        
        self._experiment.end()
        self._experiment = None
        
        return ExperimentResult(
            experiment_id=experiment_id or "unknown",
            experiment_url=experiment_url or "",
            model_type=parameters.get("model_type", "unknown"),
            hyperparameters=parameters,
            val_score=metrics.get("val_score"),
            cv_mean=metrics.get("cv_mean"),
            cv_std=metrics.get("cv_std"),
        )

    @property
    def current_experiment(self) -> Experiment | None:
        """Get current experiment."""
        return self._experiment

    @property
    def experiment_url(self) -> str | None:
        """Get current experiment URL."""
        return self._experiment.url if self._experiment else None


# Singleton tracker instance
_tracker: ExperimentTracker | None = None


def get_tracker(
    project_name: str | None = None,
    competition_name: str | None = None,
) -> ExperimentTracker:
    """Get or create experiment tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker(
            project_name=project_name,
            competition_name=competition_name,
        )
    return _tracker
