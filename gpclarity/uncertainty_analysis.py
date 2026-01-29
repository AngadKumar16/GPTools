"""
Uncertainty profiling and analysis for Gaussian Process models.
"""

from __future__ import annotations  # ✅ Postponed evaluation of annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import GPy
import numpy as np

# ✅ Only import for type checking, not at runtime
if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class UncertaintyProfiler:
    """
    Analyze and visualize uncertainty behavior across input space.

    Provides tools for computing uncertainty diagnostics, plotting
    uncertainty profiles, and identifying high/low uncertainty regions.
    """

    def __init__(self, model: GPy.models.GPRegression):
        """
        Initialize profiler with a GP model.

        Args:
            model: Trained GPy model with predict() method

        Raises:
            ValueError: If model lacks required methods
        """
        if not hasattr(model, "predict"):
            raise ValueError("Model must have predict() method")
        self.model = model

    def predict_with_uncertainty(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Safe prediction with uncertainty quantification and numerical stability.

        Args:
            X_test: Test input locations

        Returns:
            Tuple of (mean, variance) arrays

        Raises:
            RuntimeError: If prediction fails
        """
        try:
            mean, var = self.model.predict(X_test)
            # Ensure positive variance for numerical stability
            var = np.clip(var, 1e-10, None)
            return mean, var
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e

    def compute_diagnostics(self, X_test: np.ndarray) -> Dict[str, float]:
        """
        Compute spatial uncertainty metrics.

        Args:
            X_test: Input locations for analysis

        Returns:
            Dictionary of uncertainty statistics
        """
        _, var = self.predict_with_uncertainty(X_test)

        return {
            "mean_uncertainty": float(np.mean(var)),
            "max_uncertainty": float(np.max(var)),
            "uncertainty_std": float(np.std(var)),
            "total_uncertainty": float(np.sum(var)),
            "high_uncertainty_ratio": float(np.mean(var > np.percentile(var, 90))),
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
        ax: Optional["plt.Axes"] = None,  # ✅ String annotation
        confidence_level: float = 2.0,
        **plot_kwargs,
    ) -> "plt.Axes":  # ✅ String annotation
        """
        Plot uncertainty profile with optional training data overlay.

        Args:
            X_test: Test locations (shape: [n_test, n_dims])
            X_train: Training inputs (optional, shape: [n_train, n_dims])
            y_train: Training outputs (optional, shape: [n_train,])
            ax: Matplotlib axes (creates new if None)
            confidence_level: Multiplier for std dev bands (default: 2σ ≈ 95%)
            **plot_kwargs: Additional plotting arguments

        Returns:
            Matplotlib axes object

        Raises:
            ImportError: If matplotlib is not installed
        """
        # ✅ Lazy import with helpful error message
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "plot requires matplotlib. " "Install with: pip install matplotlib"
            ) from e

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
            mean_plot - confidence_level * std_plot,
            mean_plot + confidence_level * std_plot,
            alpha=plot_kwargs.get("alpha", 0.2),
            color=plot_kwargs.get("fill_color", "#1f77b4"),
            label=f"±{confidence_level}σ uncertainty",
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

        # Styling
        ax.set_xlabel("Input Space", fontsize=12)
        ax.set_ylabel("Output", fontsize=12)
        ax.set_title("Uncertainty Profile", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        return ax

    def identify_uncertainty_regions(
        self, X_test: np.ndarray, threshold_percentile: float = 90.0
    ) -> Dict[str, np.ndarray]:
        """
        Identify high and low uncertainty regions.

        Args:
            X_test: Input locations
            threshold_percentile: Percentile threshold for high uncertainty

        Returns:
            Dictionary with region masks and threshold value
        """
        _, var = self.predict_with_uncertainty(X_test)
        threshold = np.percentile(var, threshold_percentile)

        return {
            "high_uncertainty_points": X_test[var.flatten() > threshold],
            "high_uncertainty_values": var[var.flatten() > threshold],
            "low_uncertainty_points": X_test[var.flatten() <= threshold],
            "low_uncertainty_values": var[var.flatten() <= threshold],
            "threshold": float(threshold),
            "uncertainty_values": var,
            "threshold_percentile": threshold_percentile,
        }
