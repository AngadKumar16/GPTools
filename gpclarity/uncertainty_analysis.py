"""
Uncertainty profiling and analysis for Gaussian Process models.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import GPy
import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configurable parameters for uncertainty analysis."""
    min_variance: float = 1e-10
    default_confidence_level: float = 2.0
    high_uncertainty_percentile: float = 90.0
    
    def validate(self):
        """Ensure parameters are valid."""
        if self.min_variance <= 0:
            raise ValueError("min_variance must be positive")
        if self.default_confidence_level <= 0:
            raise ValueError("confidence_level must be positive")
        if not 0 < self.high_uncertainty_percentile < 100:
            raise ValueError("high_uncertainty_percentile must be in (0, 100)")
        return self


class UncertaintyProfiler:
    """
    Analyze and visualize uncertainty behavior across input space.
    """

    def __init__(
        self,
        model: GPy.models.GPRegression,
        config: Optional[UncertaintyConfig] = None
    ):
        """
        Initialize profiler with a GP model.

        Args:
            model: Trained GPy model with predict() method
            config: Uncertainty configuration (uses defaults if None)
        """
        if not hasattr(model, "predict"):
            raise ValueError("Model must have predict() method")
            
        self.model = model
        self.config = (config or UncertaintyConfig()).validate()

    def predict_with_uncertainty(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Safe prediction with uncertainty quantification.

        Args:
            X_test: Test input locations

        Returns:
            Tuple of (mean, variance) arrays
        """
        try:
            mean, var = self.model.predict(X_test)
            # Use configurable minimum variance
            var = np.clip(var, self.config.min_variance, None)
            return mean, var
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e

    def compute_diagnostics(self, X_test: np.ndarray) -> Dict[str, float]:
        """
        Compute spatial uncertainty metrics.
        """
        _, var = self.predict_with_uncertainty(X_test)

        return {
            "mean_uncertainty": float(np.mean(var)),
            "max_uncertainty": float(np.max(var)),
            "uncertainty_std": float(np.std(var)),
            "total_uncertainty": float(np.sum(var)),
            "high_uncertainty_ratio": float(
                np.mean(var > np.percentile(var, self.config.high_uncertainty_percentile))
            ),
            "median_uncertainty": float(np.median(var)),
            "uncertainty_skewness": float(self._skewness(var)),
        }

    def _skewness(self, arr: np.ndarray) -> float:
        """Compute skewness of array."""
        mean, var = np.mean(arr), np.var(arr)
        if var == 0:
            return 0.0
        return np.mean(((arr - mean) / np.sqrt(var)) ** 3)

    def plot(
        self,
        X_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        ax: Optional["plt.Axes"] = None,
        confidence_level: Optional[float] = None,
        **plot_kwargs,
    ) -> "plt.Axes":
        """
        Plot uncertainty profile.

        Args:
            X_test: Test locations
            X_train: Training inputs (optional)
            y_train: Training outputs (optional)
            ax: Matplotlib axes
            confidence_level: Multiplier for std dev bands 
                              (uses config.default_confidence_level if None)
            **plot_kwargs: Additional plotting arguments
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "plot requires matplotlib. Install with: pip install matplotlib"
            ) from e

        cfg_level = confidence_level or self.config.default_confidence_level

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        mean, var = self.predict_with_uncertainty(X_test)
        std = np.sqrt(var)

        X_plot = X_test.flatten()
        mean_plot = mean.flatten()
        std_plot = std.flatten()

        # Main prediction line
        ax.plot(
            X_plot,
            mean_plot,
            label="GP Mean",
            color=plot_kwargs.get("color", "#1f77b4"),
            linewidth=2,
            zorder=3,
        )

        # Uncertainty bands
        ax.fill_between(
            X_plot,
            mean_plot - cfg_level * std_plot,
            mean_plot + cfg_level * std_plot,
            alpha=plot_kwargs.get("alpha", 0.2),
            color=plot_kwargs.get("fill_color", "#1f77b4"),
            label=f"±{cfg_level}σ uncertainty",
            zorder=1,
        )

        # Training data scatter
        if X_train is not None and y_train is not None:
            ax.scatter(
                X_train.flatten(),
                y_train.flatten(),
                color=plot_kwargs.get("train_color", "red"),
                s=plot_kwargs.get("train_size", 50),
                zorder=5,
                label="Training Data",
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Input Space", fontsize=12)
        ax.set_ylabel("Output", fontsize=12)
        ax.set_title("Uncertainty Profile", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        return ax

    def identify_uncertainty_regions(
        self,
        X_test: np.ndarray,
        threshold_percentile: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Identify high and low uncertainty regions.

        Args:
            X_test: Input locations
            threshold_percentile: Percentile threshold 
                                  (uses config.high_uncertainty_percentile if None)
        """
        cfg_percentile = threshold_percentile or self.config.high_uncertainty_percentile
        
        _, var = self.predict_with_uncertainty(X_test)
        threshold = np.percentile(var, cfg_percentile)

        return {
            "high_uncertainty_points": X_test[var.flatten() > threshold],
            "high_uncertainty_values": var[var.flatten() > threshold],
            "low_uncertainty_points": X_test[var.flatten() <= threshold],
            "low_uncertainty_values": var[var.flatten() <= threshold],
            "threshold": float(threshold),
            "uncertainty_values": var,
            "threshold_percentile": cfg_percentile,
        }