"""
Data influence analysis: quantify how training points affect model uncertainty.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, Tuple

import GPy
import numpy as np
from scipy.linalg import LinAlgError, solve_triangular

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class InfluenceError(Exception):
    """Raised when influence computation fails."""
    pass


class DataInfluenceMap:
    """
    Compute influence scores for training data points.

    Identifies which observations most reduce model uncertainty and
    which have high leverage on predictions.
    """

    def __init__(self, model: GPy.models.GPRegression):
        """
        Initialize with GP model.

        Args:
            model: Trained GP model

        Raises:
            ValueError: If model lacks required attributes
        """
        if model is None:
            raise ValueError("Model cannot be None")
            
        if not hasattr(model, "predict"):
            raise ValueError("Model must have predict() method")
            
        if not hasattr(model, "kern"):
            raise ValueError("Model must have 'kern' attribute")
            
        self.model = model

    def compute_influence_scores(self, X_train: np.ndarray) -> np.ndarray:
        """
        Approximate influence scores using leverage scores (fast).

        Leverage score = diagonal of K(K + σ²I)⁻¹K, approximated via
        Cholesky decomposition for numerical stability.

        Args:
            X_train: Training input locations (shape: [n_train, n_dims])

        Returns:
            Influence scores array (higher = more influential)

        Raises:
            InfluenceError: If computation fails
            ValueError: If inputs are invalid
        """
        if X_train is None or not hasattr(X_train, "shape"):
            raise ValueError("X_train must be a numpy array")
            
        if X_train.shape[0] == 0:
            raise ValueError("X_train cannot be empty")

        try:
            # Compute covariance matrix
            K = self.model.kern.K(X_train, X_train)
            
            if not np.all(np.isfinite(K)):
                raise InfluenceError("Kernel matrix contains non-finite values")
                
            noise_var = float(self.model.Gaussian_noise.variance)
            
            if not np.isfinite(noise_var) or noise_var < 0:
                logger.warning(f"Invalid noise variance: {noise_var}, using fallback")
                noise_var = 1e-6
                
            K_stable = K + np.eye(K.shape[0]) * noise_var

            # Cholesky decomposition with jitter fallback
            L = None
            jitter = 1e-6
            
            try:
                L = np.linalg.cholesky(K_stable)
            except LinAlgError:
                logger.warning("Cholesky failed, adding jitter")
                for attempt in range(3):
                    try:
                        K_stable += np.eye(K.shape[0]) * jitter
                        L = np.linalg.cholesky(K_stable)
                        break
                    except LinAlgError:
                        jitter *= 10
                else:
                    raise InfluenceError("Could not decompose kernel matrix even with jitter")
                    
            if L is None:
                raise InfluenceError("Cholesky decomposition failed")

            # Solve K⁻¹ for each basis vector (leverage scores)
            n = X_train.shape[0]
            scores = np.zeros(n)

            for i in range(n):
                e_i = np.zeros(n)
                e_i[i] = 1.0

                try:
                    # Solve linear system efficiently
                    v = solve_triangular(L, e_i, lower=True)
                    inv_row = solve_triangular(L, v, lower=True, trans="T")

                    # Leverage score is inverse of diagonal element
                    diag_inv = inv_row[i]
                    
                    if not np.isfinite(diag_inv) or diag_inv == 0:
                        logger.warning(f"Invalid diagonal element at index {i}: {diag_inv}")
                        scores[i] = 0.0
                    else:
                        scores[i] = 1.0 / diag_inv
                        
                except LinAlgError as e:
                    logger.warning(f"Linear algebra error at index {i}: {e}")
                    scores[i] = 0.0

            return scores
            
        except InfluenceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error computing influence scores: {e}")
            raise InfluenceError(f"Failed to compute influence scores: {e}") from e

    def compute_loo_variance_increase(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact Leave-One-Out variance increase (slower but precise).

        Args:
            X_train: Training inputs (shape: [n_train, n_dims])
            y_train: Training outputs (shape: [n_train,])

        Returns:
            Tuple of (variance_increase, prediction_errors)

        Raises:
            InfluenceError: If computation fails
            ValueError: If inputs are invalid or shapes mismatch
        """
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None")
            
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Shape mismatch: X_train has {X_train.shape[0]} samples, "
                f"y_train has {y_train.shape[0]}"
            )
            
        if X_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")

        n = X_train.shape[0]
        variance_increase = np.zeros(n)
        prediction_errors = np.zeros(n)

        try:
            # Pre-compute full covariance for efficiency
            K_full = self.model.kern.K(X_train, X_train)
            noise_var = float(self.model.Gaussian_noise.variance)
            
            if not np.isfinite(noise_var):
                raise InfluenceError(f"Invalid noise variance: {noise_var}")
                
            np.fill_diagonal(K_full, np.diag(K_full) + noise_var)

            for i in range(n):
                # Leave-one-out indices
                idx = np.arange(n) != i

                try:
                    # Sub-covariance matrix
                    K_loo = K_full[np.ix_(idx, idx)]
                    k_star = K_full[np.ix_([i], idx)][0]

                    # Validate matrices
                    if not np.all(np.isfinite(K_loo)):
                        raise ValueError("LOO covariance matrix contains non-finite values")
                        
                    # Posterior variance at x_i without it in training
                    L = np.linalg.cholesky(K_loo)
                    v = solve_triangular(L, k_star, lower=True)
                    var_without_i = K_full[i, i] - np.dot(v, v)

                    if not np.isfinite(var_without_i) or var_without_i < 0:
                        logger.warning(f"Invalid variance without point {i}: {var_without_i}")
                        variance_increase[i] = np.nan
                        prediction_errors[i] = np.nan
                        continue

                    # Variance with x_i in training (from model)
                    mean_with_i, var_with_i = self.model.predict(X_train[i : i + 1])
                    
                    if not np.isfinite(var_with_i[0, 0]):
                        logger.warning(f"Invalid variance from model at point {i}")
                        variance_increase[i] = np.nan
                        prediction_errors[i] = np.nan
                        continue

                    variance_increase[i] = max(0, var_without_i - var_with_i[0, 0])

                    # LOO prediction error
                    if np.isfinite(mean_with_i[0, 0]) and np.isfinite(y_train[i]):
                        prediction_errors[i] = abs(mean_with_i[0, 0] - y_train[i])
                    else:
                        prediction_errors[i] = np.nan

                except (LinAlgError, ValueError) as e:
                    logger.warning(f"LOO computation failed for point {i}: {e}")
                    variance_increase[i] = np.nan
                    prediction_errors[i] = np.nan

            return variance_increase, prediction_errors
            
        except InfluenceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LOO computation: {e}")
            raise InfluenceError(f"Failed to compute LOO variance: {e}") from e

    def plot_influence(
        self,
        X_train: np.ndarray,
        influence_scores: np.ndarray,
        ax: "plt.Axes" = None,
        **scatter_kwargs,
    ) -> "plt.Axes":
        """
        Visualize data point influence.

        Args:
            X_train: Training input locations
            influence_scores: Computed influence scores
            ax: Matplotlib axes
            **scatter_kwargs: Additional scatter plot arguments

        Returns:
            Matplotlib axes object

        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If inputs are invalid
        """
        if X_train is None or influence_scores is None:
            raise ValueError("X_train and influence_scores cannot be None")
            
        if X_train.shape[0] != influence_scores.shape[0]:
            raise ValueError(
                f"Shape mismatch: X_train has {X_train.shape[0]} points, "
                f"influence_scores has {len(influence_scores)}"
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "plot_influence requires matplotlib. "
                "Install with: pip install matplotlib"
            ) from e

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Fix division by zero - use nanmax to handle NaN scores gracefully
        finite_scores = influence_scores[np.isfinite(influence_scores)]
        if len(finite_scores) == 0:
            logger.warning("No finite influence scores to plot")
            max_score = 1.0
        else:
            max_score = np.max(finite_scores)
            
        if max_score <= 0:
            max_score = 1.0

        # Replace NaN/Inf with 0 for visualization
        safe_scores = np.where(np.isfinite(influence_scores), influence_scores, 0.0)
        sizes = 100 + (safe_scores / max_score) * 400

        scatter = ax.scatter(
            X_train.flatten(),
            np.zeros_like(X_train.flatten()),
            c=safe_scores,
            s=sizes,
            cmap=scatter_kwargs.get("cmap", "viridis"),
            alpha=scatter_kwargs.get("alpha", 0.7),
            edgecolors="black",
            linewidth=1,
        )

        ax.set_xlabel("Input Space", fontsize=12)
        ax.set_title(
            "Data Point Influence Map\n(size ∝ influence)",
            fontsize=14,
            fontweight="bold",
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Influence Score", fontsize=10)

        # Remove y-axis ticks (all zero)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis="x")

        return ax

    def get_influence_report(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive influence analysis report.

        Args:
            X_train: Training inputs
            y_train: Training outputs

        Returns:
            Dictionary with influence statistics and diagnostics

        Raises:
            InfluenceError: If report generation fails
            ValueError: If inputs are invalid
        """
        try:
            scores = self.compute_influence_scores(X_train)
            loo_var, loo_err = self.compute_loo_variance_increase(X_train, y_train)

            # Find most/least influential (handling NaN values)
            finite_mask = np.isfinite(scores)
            if not np.any(finite_mask):
                raise InfluenceError("No finite influence scores available")
                
            most_inf_idx = np.nanargmax(scores)
            least_inf_idx = np.nanargmin(scores)
            
            # Compute statistics on finite values only
            finite_scores = scores[finite_mask]
            finite_loo_err = loo_err[np.isfinite(loo_err)]

            report = {
                "influence_scores": scores,
                "loo_variance_increase": loo_var,
                "loo_prediction_errors": loo_err,
                "statistics": {
                    "mean_influence": float(np.mean(finite_scores)),
                    "std_influence": float(np.std(finite_scores)),
                    "max_influence": float(np.max(finite_scores)),
                    "min_influence": float(np.min(finite_scores)),
                },
                "most_influential_point": {
                    "index": int(most_inf_idx),
                    "location": X_train[most_inf_idx].flatten(),
                    "score": float(scores[most_inf_idx]),
                    "loo_error": (
                        float(loo_err[most_inf_idx])
                        if np.isfinite(loo_err[most_inf_idx])
                        else None
                    ),
                },
                "least_influential_point": {
                    "index": int(least_inf_idx),
                    "location": X_train[least_inf_idx].flatten(),
                    "score": float(scores[least_inf_idx]),
                    "loo_error": (
                        float(loo_err[least_inf_idx])
                        if np.isfinite(loo_err[least_inf_idx])
                        else None
                    ),
                },
                "diagnostics": {
                    "high_leverage_points": int(np.sum(scores > np.percentile(finite_scores, 95))),
                    "outliers": int(np.sum(loo_err > np.percentile(finite_loo_err, 95))) if len(finite_loo_err) > 0 else 0,
                },
            }
            
            return report
            
        except InfluenceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating influence report: {e}")
            raise InfluenceError(f"Failed to generate influence report: {e}") from e