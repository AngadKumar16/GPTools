"""
Hyperparameter evolution tracking during GP model optimization.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from gpclarity.exceptions import OptimizationError, TrackingError

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class OptimizationState:
    """Snapshot of model state at a specific iteration."""
    iteration: int
    parameters: Dict[str, Union[float, np.ndarray]]
    log_likelihood: Optional[float] = None
    gradient_norm: Optional[float] = None
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class ConvergenceMetrics:
    """Statistical metrics for parameter convergence."""
    initial_mean: float
    final_mean: float
    relative_change: float
    final_std: float
    coefficient_of_variation: float
    is_converged: bool
    trend_direction: str  # 'increasing', 'decreasing', 'stable'


class HyperparameterTracker:
    """
    Track hyperparameter trajectories during model optimization.
    """

    def __init__(self, model: Any):
        """
        Initialize tracker with GP model.

        Args:
            model: GPy model or compatible object with .parameters attribute

        Raises:
            TrackingError: If model lacks required attributes
        """
        if not hasattr(model, "parameters"):
            raise TrackingError("Model must have 'parameters' attribute")
        if not hasattr(model, "optimize"):
            raise TrackingError("Model must have 'optimize' method")

        self.model = model
        self._history: List[OptimizationState] = []
        self._param_names: List[str] = [p.name for p in model.parameters]
        self._start_time: Optional[float] = None

    @property
    def history(self) -> List[OptimizationState]:
        """Get optimization history."""
        return self._history.copy()

    @property
    def iteration_count(self) -> int:
        """Get current iteration count."""
        return len(self._history)

    def record_state(self, iteration: Optional[int] = None) -> OptimizationState:
        """Snapshot current hyperparameter values."""
        params = {}
        for param in self.model.parameters:
            val = self._extract_param_value(param)
            params[param.name] = val

        # Capture optimization metadata if available
        log_likelihood = None
        gradient_norm = None

        try:
            if hasattr(self.model, "log_likelihood"):
                log_likelihood = float(self.model.log_likelihood())
            if hasattr(self.model, "gradient"):
                grad = self.model.gradient
                if grad is not None:
                    gradient_norm = float(np.linalg.norm(grad))
        except Exception as e:
            logger.debug(f"Could not extract optimization metadata: {e}")

        state = OptimizationState(
            iteration=iteration if iteration is not None else self.iteration_count,
            parameters=params,
            log_likelihood=log_likelihood,
            gradient_norm=gradient_norm,
        )

        self._history.append(state)
        return state

    @staticmethod
    def _extract_param_value(param: Any) -> Union[float, np.ndarray]:
        """Safely extract scalar or array value from GPy parameter."""
        val = param.param_array

        if val is None:
            return 0.0

        arr = np.atleast_1d(val)

        if len(arr) == 1:
            return float(arr[0])
        else:
            return arr.copy()

    def wrapped_optimize(
        self,
        max_iters: int = 100,
        callback: Optional[Callable[[Any, int, List[OptimizationState]], None]] = None,
        capture_every: int = 1,
        convergence_tolerance: float = 1e-6,
        patience: int = 10,
        **optimize_kwargs,
    ) -> List[OptimizationState]:
        """
        Perform optimization with intelligent tracking and early stopping.
        """
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}")

        self._history = []
        self._start_time = time.perf_counter()
        best_ll = -np.inf
        patience_counter = 0

        logger.info(f"Starting tracked optimization: max_iters={max_iters}")

        for i in range(max_iters):
            try:
                # Single iteration optimization
                self.model.optimize(max_iters=1, **optimize_kwargs)

                # Record state based on capture frequency
                if i % capture_every == 0:
                    state = self.record_state(iteration=i)

                    # Early stopping check based on log-likelihood
                    if state.log_likelihood is not None:
                        if state.log_likelihood > best_ll + convergence_tolerance:
                            best_ll = state.log_likelihood
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            logger.info(
                                f"Early stopping at iteration {i} "
                                f"(no improvement for {patience} steps)"
                            )
                            break

                # User callback
                if callback:
                    try:
                        callback(self.model, i, self._history)
                    except Exception as e:
                        logger.warning(f"User callback failed at iteration {i}: {e}")

            except KeyboardInterrupt:
                logger.info(f"Optimization interrupted by user at iteration {i}")
                break
            except Exception as e:
                logger.error(f"Optimization failed at iteration {i}: {e}")
                raise OptimizationError(f"Optimization failed at iteration {i}: {e}") from e

        total_time = time.perf_counter() - self._start_time
        logger.info(
            f"Optimization complete: {self.iteration_count} iterations "
            f"in {total_time:.2f}s"
        )

        return self._history

    def get_parameter_trajectory(self, param_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time-series data for a specific parameter.

        Args:
            param_name: Name of parameter to extract

        Returns:
            Tuple of (iterations, values)

        Raises:
            KeyError: If parameter not found
        """
        if not self._history:
            raise TrackingError("No optimization history recorded")

        if param_name not in self._param_names:
            raise KeyError(
                f"Unknown parameter: {param_name}. " f"Available: {self._param_names}"
            )

        iterations = np.array([s.iteration for s in self._history])
        values = np.array([s.parameters[param_name] for s in self._history])

        return iterations, values

    def get_convergence_report(
        self,
        window: int = 10,
        convergence_threshold: float = 0.01,
    ) -> Dict[str, ConvergenceMetrics]:
        """
        Analyze parameter convergence with statistical rigor.

        Args:
            window: Number of iterations for convergence analysis
            convergence_threshold: CV threshold for convergence detection

        Returns:
            Dictionary mapping parameter names to ConvergenceMetrics
        """
        if len(self._history) < window * 2:
            raise TrackingError(
                f"Need at least {window * 2} iterations for convergence analysis, "
                f"got {len(self._history)}"
            )

        self._validate_convergence_window(window, len(self._history))

        report = {}

        for param_name in self._param_names:
            # Extract values handling multi-dimensional parameters
            values = np.array([s.parameters[param_name] for s in self._history])

            # Flatten multi-dimensional for scalar metrics
            if values.ndim > 1:
                values = np.linalg.norm(values, axis=1)

            recent = values[-window:]
            earlier = values[:window]

            initial_mean = float(np.mean(earlier))
            final_mean = float(np.mean(recent))
            final_std = float(np.std(recent))

            # Avoid division by zero
            denom = abs(initial_mean) if initial_mean != 0 else 1e-10
            relative_change = abs(final_mean - initial_mean) / denom

            cv = final_std / (abs(final_mean) + 1e-10)
            is_converged = cv < convergence_threshold

            # Trend detection using linear regression
            if len(values) >= 3:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                if abs(slope) < 1e-10:
                    trend = "stable"
                elif slope > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
            else:
                trend = "unknown"

            report[param_name] = ConvergenceMetrics(
                initial_mean=initial_mean,
                final_mean=final_mean,
                relative_change=relative_change,
                final_std=final_std,
                coefficient_of_variation=cv,
                is_converged=is_converged,
                trend_direction=trend,
            )

        return report

    @staticmethod
    def _validate_convergence_window(window: int, history_length: int) -> None:
        """Validate window size for convergence analysis."""
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        if window > history_length // 2:
            raise ValueError(
                f"Window ({window}) too large for history length ({history_length}). "
                f"Max allowed: {history_length // 2}"
            )

    def detect_optimization_issues(self) -> Dict[str, Any]:
        """Detect common optimization problems."""
        issues = {"warnings": [], "recommendations": [], "metrics": {}}

        if not self._history:
            issues["warnings"].append("No optimization history available")
            return issues

        # Check for NaN/Inf in parameters
        for state in self._history:
            for name, val in state.parameters.items():
                arr = np.atleast_1d(val)
                if not np.all(np.isfinite(arr)):
                    issues["warnings"].append(
                        f"Non-finite values detected in {name} at iteration {state.iteration}"
                    )

        # Check for oscillation
        if len(self._history) >= 20:
            for param_name in self._param_names:
                _, values = self.get_parameter_trajectory(param_name)
                recent_std = np.std(values[-10:])
                overall_range = np.max(values) - np.min(values)

                if overall_range > 0 and recent_std / overall_range > 0.3:
                    issues["warnings"].append(
                        f"{param_name} showing high oscillation (no convergence)"
                    )
                    issues["recommendations"].append(
                        f"Consider reducing learning rate for {param_name}"
                    )

        # Log-likelihood trends
        lls = [
            s.log_likelihood for s in self._history if s.log_likelihood is not None
        ]
        if len(lls) >= 2:
            if lls[-1] < lls[0]:
                issues["warnings"].append("Log-likelihood decreased during optimization")
                issues["recommendations"].append("Check for numerical instability")

            issues["metrics"]["ll_improvement"] = lls[-1] - lls[0]
            issues["metrics"]["final_ll"] = lls[-1]

        return issues

    def plot_evolution(
        self,
        params: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show_convergence: bool = True,
        show_ll: bool = True,
        n_cols: int = 2,
    ) -> "plt.Figure":
        """
        Plot parameter trajectories with optional convergence indicators.
        """
        from gpclarity.plotting import plot_optimization_trajectory

        if not self._history:
            raise TrackingError(
                "No optimization history recorded. " "Run wrapped_optimize() first."
            )

        return plot_optimization_trajectory(
            self,
            params=params,
            figsize=figsize,
            show_convergence=show_convergence,
            show_ll=show_ll,
            n_cols=n_cols,
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """Export history to pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "to_dataframe() requires pandas. Install: pip install pandas"
            ) from e

        records = []
        for state in self._history:
            record = {
                "iteration": state.iteration,
                "log_likelihood": state.log_likelihood,
                "gradient_norm": state.gradient_norm,
                "timestamp": state.timestamp,
            }
            # Flatten parameters
            for name, val in state.parameters.items():
                arr = np.atleast_1d(val)
                if len(arr) == 1:
                    record[name] = float(arr[0])
                else:
                    for i, v in enumerate(arr):
                        record[f"{name}_{i}"] = float(v)
            records.append(record)

        return pd.DataFrame(records)

    def __len__(self) -> int:
        """Return number of recorded states."""
        return len(self._history)

    def __repr__(self) -> str:
        """String representation."""
        status = f"{self.iteration_count} iterations recorded"
        if self._history:
            duration = self._history[-1].timestamp - self._history[0].timestamp
            status += f", {duration:.2f}s duration"
        return f"<HyperparameterTracker: {status}>"