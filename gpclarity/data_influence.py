"""
Data influence analysis: quantify how training points affect model uncertainty.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import GPy
import numpy as np

from gpclarity.exceptions import InfluenceError
from gpclarity.utils import _cholesky_with_jitter, _validate_kernel_matrix

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InfluenceResult:
    """Container for influence computation results."""
    scores: np.ndarray
    method: str
    computation_time: float
    n_points: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __array__(self):
        """Allow numpy operations on result directly."""
        return self.scores


def _validate_train_data(func: Callable) -> Callable:
    """Decorator to standardize training data validation."""
    def wrapper(
        self, 
        X_train: np.ndarray, 
        y_train: Optional[np.ndarray] = None,
        *args, 
        **kwargs
    ):
        if X_train is None:
            raise ValueError("X_train cannot be None")
            
        if not hasattr(X_train, "shape"):
            raise ValueError("X_train must be array-like with shape attribute")
            
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
            
        if X_train.shape[0] == 0:
            raise ValueError("X_train cannot be empty")
            
        if y_train is not None:
            if y_train.shape[0] != X_train.shape[0]:
                raise ValueError(
                    f"Shape mismatch: X_train has {X_train.shape[0]} samples, "
                    f"y_train has {y_train.shape[0]}"
                )
            if y_train.ndim != 1 and (y_train.ndim != 2 or y_train.shape[1] != 1):
                warnings.warn(
                    f"y_train should be 1D, got shape {y_train.shape}. "
                    "Flattening automatically.",
                    UserWarning
                )
                y_train = y_train.ravel()
                
        return func(self, X_train, y_train, *args, **kwargs)
    
    return wrapper


class DataInfluenceMap:
    """
    Compute influence scores for training data points.

    Identifies which observations most reduce model uncertainty and
    which have high leverage on predictions.

    Attributes:
        model: Trained GP model
        _cache: Internal cache for expensive computations
    """

    def __init__(self, model: GPy.models.GPRegression):
        """
        Initialize with GP model.

        Args:
            model: Trained GP model with 'predict' and 'kern' attributes

        Raises:
            ValueError: If model lacks required attributes
        """
        if not hasattr(model, "predict"):
            raise ValueError("Model must have predict() method")
        if not hasattr(model, "kern"):
            raise ValueError("Model must have 'kern' attribute")
            
        self.model = model
        self._cache: Dict[str, Any] = {}
        self._cache_key: Optional[str] = None

    def _get_cache_key(self, X: np.ndarray) -> str:
        """Generate unique cache key for input data."""
        return f"{X.shape}_{hash(X.tobytes()) % (2**32)}"

    def _get_cached_kernel(
        self, 
        X: np.ndarray, 
        noise_var: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Retrieve or compute kernel matrix with Cholesky decomposition.
        
        Returns:
            Tuple of (K, L, cache_key)
        """
        cache_key = self._get_cache_key(X)
        
        if cache_key == self._cache_key and "K" in self._cache:
            logger.debug("Using cached kernel matrix")
            K = self._cache["K"]
            L = self._cache["L"]
        else:
            logger.debug("Computing new kernel matrix")
            K = self.model.kern.K(X, X)
            _validate_kernel_matrix(K)
            
            if noise_var is None:
                noise_var = float(self.model.Gaussian_noise.variance)
                
            if not np.isfinite(noise_var) or noise_var < 0:
                logger.warning(f"Invalid noise variance: {noise_var}, using fallback")
                noise_var = 1e-6
                
            K_stable = K + np.eye(K.shape[0]) * noise_var
            L = _cholesky_with_jitter(K_stable)
            
            # Update cache
            self._cache = {"K": K, "L": L, "noise_var": noise_var}
            self._cache_key = cache_key
            
        return K, L, cache_key

    def clear_cache(self) -> None:
        """Clear internal computation cache to free memory."""
        self._cache.clear()
        self._cache_key = None
        logger.debug("Cache cleared")

    @_validate_train_data
    def compute_influence_scores(
        self, 
        X_train: np.ndarray,
        *,
        use_cache: bool = True
    ) -> InfluenceResult:
        """
        Compute influence scores using leverage scores (optimized O(n³)).

        Leverage scores computed via diagonal of hat matrix using cached
        Cholesky decomposition.

        Args:
            X_train: Training input locations (shape: [n_train, n_dims])
            use_cache: Whether to use internal cache for kernel matrix

        Returns:
            InfluenceResult with scores and metadata

        Raises:
            InfluenceError: If computation fails
        """
        start_time = time.perf_counter()
        
        try:
            # Get or compute kernel and Cholesky factor
            if use_cache:
                K, L, _ = self._get_cached_kernel(X_train)
            else:
                K = self.model.kern.K(X_train, X_train)
                _validate_kernel_matrix(K)
                noise_var = float(self.model.Gaussian_noise.variance)
                K_stable = K + np.eye(K.shape[0]) * max(noise_var, 1e-6)
                L = _cholesky_with_jitter(K_stable)

            n = X_train.shape[0]
            
            # Optimized leverage score computation: O(n³) instead of O(n⁴)
            # L @ L.T = K, solve for L_inv then K_inv = L_inv.T @ L_inv
            L_inv = np.linalg.solve_triangular(L, np.eye(n), lower=True)
            K_inv = L_inv.T @ L_inv
            scores = 1.0 / np.diag(K_inv)
            
            # Handle numerical edge cases
            if not np.all(np.isfinite(scores)):
                n_invalid = np.sum(~np.isfinite(scores))
                logger.warning(f"{n_invalid} influence scores are non-finite, clipping to 0")
                scores = np.where(np.isfinite(scores), scores, 0.0)
                
            if np.any(scores < 0):
                n_neg = np.sum(scores < 0)
                logger.warning(f"{n_neg} negative influence scores found, setting to 0")
                scores = np.maximum(scores, 0)

            computation_time = time.perf_counter() - start_time
            
            return InfluenceResult(
                scores=scores,
                method="leverage",
                computation_time=computation_time,
                n_points=n,
                metadata={
                    "kernel_type": type(self.model.kern).__name__,
                    "noise_variance": float(self.model.Gaussian_noise.variance),
                    "cache_used": use_cache,
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to compute influence scores: {e}")
            raise InfluenceError(f"Influence computation failed: {e}") from e

    @_validate_train_data
    def compute_loo_variance_increase(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact Leave-One-Out variance increase with optional parallelization.

        Args:
            X_train: Training inputs (shape: [n_train, n_dims])
            y_train: Training outputs (shape: [n_train,] or [n_train, 1])
            n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
            verbose: Whether to display progress bar

        Returns:
            Tuple of (variance_increase, prediction_errors)

        Raises:
            InfluenceError: If computation fails
        """
        n = X_train.shape[0]
        variance_increase = np.full(n, np.nan)
        prediction_errors = np.full(n, np.nan)

        try:
            # Pre-compute full covariance once
            K_full = self.model.kern.K(X_train, X_train)
            _validate_kernel_matrix(K_full)
            
            noise_var = float(self.model.Gaussian_noise.variance)
            if not np.isfinite(noise_var):
                raise InfluenceError(f"Invalid noise variance: {noise_var}")
                
            K_full = K_full.copy()
            np.fill_diagonal(K_full, np.diag(K_full) + noise_var)

            # Setup iterator with optional progress bar
            iterator = range(n)
            if verbose:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, desc="Computing LOO")
                except ImportError:
                    logger.warning("tqdm not installed, progress bar disabled")
                    verbose = False

            # Parallel or sequential execution
            if n_jobs != 1:
                result = self._compute_loo_parallel(
                    X_train, y_train, K_full, n_jobs, iterator
                )
                variance_increase, prediction_errors = result
            else:
                for i in iterator:
                    result = self._compute_loo_point(
                        i, X_train, y_train, K_full
                    )
                    variance_increase[i], prediction_errors[i] = result

            return variance_increase, prediction_errors
            
        except Exception as e:
            logger.error(f"Failed to compute LOO variance: {e}")
            raise InfluenceError(f"LOO computation failed: {e}") from e

    def _compute_loo_point(
        self,
        i: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        K_full: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute LOO metrics for single point."""
        n = X_train.shape[0]
        idx = np.arange(n) != i
        
        try:
            # Extract sub-matrices
            K_loo = K_full[np.ix_(idx, idx)]
            k_star = K_full[np.ix_([i], idx)][0]
            
            # Quick validation
            if np.any(~np.isfinite(K_loo)) or np.any(~np.isfinite(k_star)):
                logger.warning(f"Non-finite values in LOO matrices for point {i}")
                return np.nan, np.nan
            
            # Cholesky with jitter if needed
            L = _cholesky_with_jitter(K_loo)
            
            # Solve for variance without point i
            v = np.linalg.solve_triangular(L, k_star, lower=True)
            k_inv_k = np.dot(v, v)
            var_without_i = K_full[i, i] - k_inv_k
            
            if var_without_i < 0:
                logger.debug(f"Negative variance at point {i}: {var_without_i}, clamping to 0")
                var_without_i = 0
                
            # Get model prediction with point i
            mean_with_i, var_with_i = self.model.predict(X_train[i:i+1])
            
            var_increase = max(0, var_without_i - float(var_with_i[0, 0]))
            
            # Prediction error
            pred_error = abs(float(mean_with_i[0, 0]) - float(y_train[i]))
            
            return var_increase, pred_error
            
        except Exception as e:
            logger.debug(f"LOO failed for point {i}: {e}")
            return np.nan, np.nan

    def _compute_loo_parallel(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        K_full: np.ndarray,
        n_jobs: int,
        iterator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel LOO computation using joblib."""
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warnings.warn("joblib not installed, falling back to sequential")
            return self._compute_loo_variance_increase(
                X_train, y_train, n_jobs=1, verbose=False
            )

        n = X_train.shape[0]
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._compute_loo_point)(i, X_train, y_train, K_full)
            for i in iterator
        )
        
        variance_increase = np.array([r[0] for r in results])
        prediction_errors = np.array([r[1] for r in results])
        return variance_increase, prediction_errors

    def plot_influence(
        self,
        X_train: np.ndarray,
        influence_scores: Union[np.ndarray, InfluenceResult],
        ax: Optional["plt.Axes"] = None,
        **scatter_kwargs,
    ) -> "plt.Axes":
        """
        Visualize data point influence (delegated to plotting module).

        Args:
            X_train: Training input locations
            influence_scores: Computed scores or InfluenceResult
            ax: Matplotlib axes (created if None)
            **scatter_kwargs: Additional scatter arguments

        Returns:
            Matplotlib axes object

        Raises:
            ImportError: If matplotlib not installed
            ValueError: If input dimensions > 2
        """
        # Deferred import to keep computation module lightweight
        from gpclarity.plotting import plot_influence_map
        
        if isinstance(influence_scores, InfluenceResult):
            influence_scores = influence_scores.scores
            
        return plot_influence_map(
            X_train, influence_scores, ax=ax, **scatter_kwargs
        )

    @_validate_train_data
    def get_influence_report(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        *,
        compute_loo: bool = True,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Comprehensive influence analysis report.

        Args:
            X_train: Training inputs
            y_train: Training outputs
            compute_loo: Whether to include LOO analysis (slow for large n)
            n_jobs: Parallel jobs for LOO computation

        Returns:
            Dictionary with influence statistics and diagnostics
        """
        start_time = time.perf_counter()
        
        # Leverage scores (fast)
        leverage_result = self.compute_influence_scores(X_train)
        scores = leverage_result.scores
        
        # LOO analysis (optional, slower)
        loo_var, loo_err = None, None
        if compute_loo:
            try:
                loo_var, loo_err = self.compute_loo_variance_increase(
                    X_train, y_train, n_jobs=n_jobs
                )
            except Exception as e:
                logger.warning(f"LOO computation skipped: {e}")
        
        # Compute statistics on finite values
        finite_mask = np.isfinite(scores)
        finite_scores = scores[finite_mask]
        
        if len(finite_scores) == 0:
            raise InfluenceError("No finite influence scores available")
        
        # Percentile-based diagnostics
        p95 = np.percentile(finite_scores, 95)
        p5 = np.percentile(finite_scores, 5)
        
        most_inf_idx = int(np.nanargmax(scores))
        least_inf_idx = int(np.nanargmin(scores))
        
        report = {
            "computation_summary": {
                "total_time": time.perf_counter() - start_time,
                "leverage_time": leverage_result.computation_time,
                "n_points": leverage_result.n_points,
                "method": leverage_result.method,
            },
            "influence_scores": {
                "mean": float(np.mean(finite_scores)),
                "std": float(np.std(finite_scores)),
                "median": float(np.median(finite_scores)),
                "max": float(np.max(finite_scores)),
                "min": float(np.min(finite_scores)),
                "p95": float(p95),
                "p5": float(p5),
            },
            "most_influential": {
                "index": most_inf_idx,
                "location": X_train[most_inf_idx].tolist(),
                "score": float(scores[most_inf_idx]),
            },
            "least_influential": {
                "index": least_inf_idx,
                "location": X_train[least_inf_idx].tolist(),
                "score": float(scores[least_inf_idx]),
            },
            "diagnostics": {
                "high_leverage_count": int(np.sum(scores > p95)),
                "low_influence_count": int(np.sum(scores < p5)),
                "non_finite_scores": int(np.sum(~finite_mask)),
            },
        }
        
        # Add LOO results if available
        if loo_var is not None and loo_err is not None:
            finite_loo = loo_err[np.isfinite(loo_err)]
            report["loo_analysis"] = {
                "variance_increase": loo_var.tolist(),
                "prediction_errors": loo_err.tolist(),
                "mean_error": float(np.mean(finite_loo)) if len(finite_loo) > 0 else None,
                "max_error": float(np.max(finite_loo)) if len(finite_loo) > 0 else None,
            }
            
        return report