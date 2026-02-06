"""
Uncertainty profiling and analysis for Gaussian Process models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
from gpclarity.exceptions import UncertaintyError
from gpclarity.utils import _validate_array

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class UncertaintyRegion(Enum):
    """Classification of uncertainty regions."""
    EXTRAPOLATION = auto()  # Far from training data
    INTERPOLATION = auto()  # Within training data convex hull
    BOUNDARY = auto()  # Near edge of training data
    HIGH_NOISE = auto()  # High aleatoric uncertainty
    STRUCTURAL = auto()  # High epistemic uncertainty


@dataclass(frozen=True)
class UncertaintyConfig:
    """Configurable parameters for uncertainty analysis."""
    min_variance: float = 1e-10
    max_variance: float = 1e10
    default_confidence_level: float = 2.0
    high_uncertainty_percentile: float = 90.0
    calibration_bins: int = 10
    numerical_jitter: float = 1e-9
    
    def __post_init__(self):
        if self.min_variance <= 0:
            raise ValueError("min_variance must be positive")
        if self.max_variance <= self.min_variance:
            raise ValueError("max_variance must be > min_variance")
        if self.default_confidence_level <= 0:
            raise ValueError("confidence_level must be positive")
        if not 0 < self.high_uncertainty_percentile < 100:
            raise ValueError("high_uncertainty_percentile must be in (0, 100)")


@dataclass
class UncertaintyDiagnostics:
    """Comprehensive uncertainty diagnostics."""
    mean_uncertainty: float
    median_uncertainty: float
    max_uncertainty: float
    min_uncertainty: float
    std_uncertainty: float
    total_uncertainty: float
    high_uncertainty_ratio: float
    uncertainty_skewness: float
    uncertainty_kurtosis: float
    coefficient_of_variation: float
    
    # Spatial characteristics
    n_extrapolation_points: int = 0
    n_boundary_points: int = 0
    uncertainty_gradient_mean: float = 0.0
    
    @property
    def is_well_calibrated(self) -> bool:
        """Check if uncertainty distribution looks reasonable."""
        # Heuristic: CV should be moderate, not extreme
        return 0.1 < self.coefficient_of_variation < 10.0


@dataclass
class PredictionResult:
    """Container for prediction with uncertainty."""
    mean: np.ndarray
    variance: np.ndarray
    std: np.ndarray
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.confidence_intervals is None:
            self.confidence_intervals = {}
    
    def get_interval(self, level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence interval at specified level."""
        key = f"{level:.1f}sigma"
        if key not in self.confidence_intervals:
            lower = self.mean - level * self.std
            upper = self.mean + level * self.std
            self.confidence_intervals[key] = (lower, upper)
        return self.confidence_intervals[key]


class UncertaintyQuantifier(Protocol):
    """Protocol for custom uncertainty quantification methods."""
    def __call__(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        X_train: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


class UncertaintyProfiler:
    """
    Analyze and visualize uncertainty behavior across input space.
    
    Provides comprehensive uncertainty quantification including:
    - Point predictions with calibrated intervals
    - Spatial uncertainty analysis
    - Extrapolation detection
    - Uncertainty decomposition (aleatoric vs epistemic)
    """

    def __init__(
        self,
        model: Any,
        config: Optional[UncertaintyConfig] = None,
        X_train: Optional[np.ndarray] = None,
    ):
        """
        Initialize profiler with a GP model.

        Args:
            model: Trained GP model with predict() method
            config: Uncertainty configuration
            X_train: Training data for extrapolation detection (optional)
        """
        if not hasattr(model, "predict"):
            raise UncertaintyError("Model must have predict() method")
        
        self.model = model
        self.config = config or UncertaintyConfig()
        self.X_train = X_train
        
        # Cache for expensive computations
        self._prediction_cache: Dict[str, PredictionResult] = {}
        self._diagnostics_cache: Optional[UncertaintyDiagnostics] = None

    def predict(
        self,
        X_test: np.ndarray,
        *,
        return_covariance: bool = False,
        cache_key: Optional[str] = None,
    ) -> PredictionResult:
        """
        Safe prediction with comprehensive uncertainty quantification.

        Args:
            X_test: Test input locations (n_test, n_features)
            return_covariance: If True, also return full covariance matrix
            cache_key: Optional key for caching results

        Returns:
            PredictionResult with mean, variance, std, and intervals

        Raises:
            UncertaintyError: If prediction fails
        """
        # Check cache
        if cache_key and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        X_test = _validate_array(X_test, "X_test")
        
        try:
            # Handle different model interfaces
            if return_covariance and hasattr(self.model, 'predict_full_cov'):
                mean, var = self.model.predict_full_cov(X_test)
                var = np.diag(var)  # Extract diagonal for consistency
            else:
                mean, var = self.model.predict(X_test)
            
            # Ensure correct shapes
            mean = np.atleast_1d(mean).flatten()
            var = np.atleast_1d(var).flatten()
            
            # Numerical safety
            var = np.clip(
                var, 
                self.config.min_variance, 
                self.config.max_variance
            )
            
            # Check for issues
            if not np.all(np.isfinite(mean)):
                n_invalid = np.sum(~np.isfinite(mean))
                logger.warning(f"{n_invalid} predictions are non-finite")
            
            result = PredictionResult(
                mean=mean,
                variance=var,
                std=np.sqrt(var),
            )
            
            # Cache if requested
            if cache_key:
                self._prediction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            raise UncertaintyError(f"Prediction failed: {e}") from e

    def compute_diagnostics(
        self,
        X_test: np.ndarray,
        force_recompute: bool = False,
    ) -> UncertaintyDiagnostics:
        """
        Compute comprehensive spatial uncertainty metrics.

        Args:
            X_test: Test input locations
            force_recompute: Ignore cache if True

        Returns:
            UncertaintyDiagnostics with detailed statistics
        """
        if not force_recompute and self._diagnostics_cache is not None:
            return self._diagnostics_cache
        
        pred = self.predict(X_test)
        var = pred.variance
        
        # Basic statistics
        mean_var = float(np.mean(var))
        median_var = float(np.median(var))
        max_var = float(np.max(var))
        min_var = float(np.min(var))
        std_var = float(np.std(var))
        
        # Advanced statistics
        skew = self._compute_skewness(var)
        kurt = self._compute_kurtosis(var)
        cv = std_var / mean_var if mean_var > 0 else 0.0
        
        # High uncertainty ratio
        threshold = np.percentile(var, self.config.high_uncertainty_percentile)
        high_unc_ratio = float(np.mean(var > threshold))
        
        # Spatial analysis
        n_ext, n_bound = 0, 0
        grad_mean = 0.0
        
        if self.X_train is not None:
            regions = self.classify_regions(X_test)
            n_ext = np.sum(regions == UncertaintyRegion.EXTRAPOLATION)
            n_bound = np.sum(regions == UncertaintyRegion.BOUNDARY)
            
            # Uncertainty gradient (how fast uncertainty changes)
            if X_test.shape[0] > 1 and X_test.shape[1] == 1:
                sorted_idx = np.argsort(X_test.flatten())
                var_sorted = var[sorted_idx]
                gradients = np.abs(np.diff(var_sorted))
                grad_mean = float(np.mean(gradients)) if len(gradients) > 0 else 0.0
        
        diagnostics = UncertaintyDiagnostics(
            mean_uncertainty=mean_var,
            median_uncertainty=median_var,
            max_uncertainty=max_var,
            min_uncertainty=min_var,
            std_uncertainty=std_var,
            total_uncertainty=float(np.sum(var)),
            high_uncertainty_ratio=high_unc_ratio,
            uncertainty_skewness=skew,
            uncertainty_kurtosis=kurt,
            coefficient_of_variation=cv,
            n_extrapolation_points=n_ext,
            n_boundary_points=n_bound,
            uncertainty_gradient_mean=grad_mean,
        )
        
        self._diagnostics_cache = diagnostics
        return diagnostics

    def classify_regions(
        self,
        X_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Classify test points by uncertainty region type.
        
        Args:
            X_test: Test locations
            X_train: Training data (uses self.X_train if None)

        Returns:
            Array of UncertaintyRegion enums
        """
        X_train = X_train or self.X_train
        if X_train is None:
            logger.warning("No training data provided, all points marked as extrapolation")
            return np.full(X_test.shape[0], UncertaintyRegion.EXTRAPOLATION)
        
        X_test = _validate_array(X_test, "X_test")
        X_train = _validate_array(X_train, "X_train")
        
        regions = np.full(X_test.shape[0], UncertaintyRegion.INTERPOLATION, dtype=object)
        
        # Distance to nearest training point
        distances = self._compute_distances_to_train(X_test, X_train)
        
        # Training data hull (convex hull for 2D+, range for 1D)
        if X_train.shape[1] == 1:
            # 1D: check if within training range
            train_min, train_max = np.min(X_train), np.max(X_train)
            tolerance = 0.05 * (train_max - train_min)
            
            for i, x in enumerate(X_test):
                if x < train_min - tolerance or x > train_max + tolerance:
                    regions[i] = UncertaintyRegion.EXTRAPOLATION
                elif x < train_min + tolerance or x > train_max - tolerance:
                    regions[i] = UncertaintyRegion.BOUNDARY
        else:
            # Multi-dimensional: use distance-based heuristic
            train_diameter = np.max(distances)
            threshold = train_diameter * 0.5
            
            for i, d in enumerate(distances):
                if d > threshold:
                    regions[i] = UncertaintyRegion.EXTRAPOLATION
                elif d < train_diameter * 0.1:
                    regions[i] = UncertaintyRegion.BOUNDARY
        
        # Refine with uncertainty magnitude
        pred = self.predict(X_test)
        high_unc_mask = pred.variance > np.percentile(pred.variance, 75)
        
        for i in np.where(high_unc_mask)[0]:
            if regions[i] == UncertaintyRegion.INTERPOLATION:
                regions[i] = UncertaintyRegion.STRUCTURAL
        
        return regions

    def identify_uncertainty_regions(
        self,
        X_test: np.ndarray,
        threshold_percentile: Optional[float] = None,
        return_regions: bool = False,
    ) -> Dict[str, Any]:
        """
        Identify and characterize high/low uncertainty regions.

        Args:
            X_test: Input locations
            threshold_percentile: Percentile threshold for "high" uncertainty
            return_regions: If True, include full region classification

        Returns:
            Dictionary with region characteristics and statistics
        """
        percentile = threshold_percentile or self.config.high_uncertainty_percentile
        
        pred = self.predict(X_test)
        var = pred.variance
        threshold = np.percentile(var, percentile)
        
        high_unc_mask = var.flatten() > threshold
        low_unc_mask = ~high_unc_mask
        
        result = {
            "high_uncertainty": {
                "points": X_test[high_unc_mask],
                "values": var[high_unc_mask],
                "indices": np.where(high_unc_mask)[0],
                "mean_uncertainty": float(np.mean(var[high_unc_mask])) if np.any(high_unc_mask) else 0.0,
            },
            "low_uncertainty": {
                "points": X_test[low_unc_mask],
                "values": var[low_unc_mask],
                "indices": np.where(low_unc_mask)[0],
                "mean_uncertainty": float(np.mean(var[low_unc_mask])) if np.any(low_unc_mask) else 0.0,
            },
            "threshold": float(threshold),
            "threshold_percentile": percentile,
            "total_points": X_test.shape[0],
            "high_uncertainty_ratio": float(np.mean(high_unc_mask)),
        }
        
        if return_regions:
            regions = self.classify_regions(X_test)
            result["region_breakdown"] = {
                region.name: int(np.sum(regions == region))
                for region in UncertaintyRegion
            }
            result["regions"] = regions
        
        return result

    def calibrate_uncertainty(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "scaling",
    ) -> Dict[str, float]:
        """
        Calibrate uncertainty using validation data.
        
        Args:
            X_val: Validation inputs
            y_val: Validation targets
            method: Calibration method ('scaling', 'isotonic', 'none')

        Returns:
            Calibration parameters and metrics
        """
        pred = self.predict(X_val)
        residuals = np.abs(y_val.flatten() - pred.mean)
        coverage = residuals < (self.config.default_confidence_level * pred.std)
        
        empirical_coverage = np.mean(coverage)
        target_coverage = 0.95  # For 2-sigma
        
        if method == "scaling":
            # Find optimal sigma scaling
            from scipy.optimize import minimize_scalar
            
            def objective(scale):
                scaled_std = pred.std * scale
                cov = np.mean(residuals < (self.config.default_confidence_level * scaled_std))
                return (cov - target_coverage) ** 2
            
            result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
            optimal_scale = result.x
            
            return {
                "method": "scaling",
                "optimal_scale": float(optimal_scale),
                "original_coverage": float(empirical_coverage),
                "target_coverage": target_coverage,
                "miscalibration": float(abs(empirical_coverage - target_coverage)),
            }
        
        return {
            "method": "none",
            "empirical_coverage": float(empirical_coverage),
            "target_coverage": target_coverage,
        }

    def plot(
        self,
        X_test: np.ndarray,
        *,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        ax: Optional["plt.Axes"] = None,
        confidence_levels: Tuple[float, ...] = (1.0, 2.0),
        plot_std: bool = False,
        fill_alpha: float = 0.2,
        color_mean: str = "#1f77b4",
        color_fill: str = "#1f77b4",
        color_train: str = "red",
        show_regions: bool = False,
        **kwargs,
    ) -> "plt.Axes":
        """
        Comprehensive uncertainty visualization.
        
        Delegates to plotting module for rendering.
        """
        from gpclarity.plotting import plot_uncertainty_profile
        
        X_train = X_train or self.X_train
        
        return plot_uncertainty_profile(
            self,
            X_test,
            X_train=X_train,
            y_train=y_train,
            y_test=y_test,
            ax=ax,
            confidence_levels=confidence_levels,
            plot_std=plot_std,
            fill_alpha=fill_alpha,
            color_mean=color_mean,
            color_fill=color_fill,
            color_train=color_train,
            show_regions=show_regions,
            **kwargs,
        )

    def get_summary(self, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Generate comprehensive uncertainty summary report.
        
        Args:
            X_test: Test locations for analysis

        Returns:
            Dictionary with summary statistics and recommendations
        """
        diagnostics = self.compute_diagnostics(X_test)
        regions = self.identify_uncertainty_regions(X_test, return_regions=True)
        
        # Generate recommendations
        recommendations = []
        
        if diagnostics.coefficient_of_variation > 5.0:
            recommendations.append(
                "High uncertainty variance: consider adaptive sampling"
            )
        
        if regions["high_uncertainty_ratio"] > 0.5:
            recommendations.append(
                "More than 50% high uncertainty: model needs more data"
            )
        
        if diagnostics.n_extrapolation_points > 0:
            recommendations.append(
                f"{diagnostics.n_extrapolation_points} extrapolation points: "
                "predictions unreliable in these regions"
            )
        
        if not diagnostics.is_well_calibrated:
            recommendations.append(
                "Uncertainty distribution unusual: check model specification"
            )
        
        return {
            "diagnostics": {
                "mean_uncertainty": diagnostics.mean_uncertainty,
                "uncertainty_range": [
                    diagnostics.min_uncertainty,
                    diagnostics.max_uncertainty,
                ],
                "high_uncertainty_ratio": diagnostics.high_uncertainty_ratio,
                "extrapolation_points": diagnostics.n_extrapolation_points,
            },
            "regions": regions.get("region_breakdown", {}),
            "recommendations": recommendations,
            "well_specified": diagnostics.is_well_calibrated,
        }

    def clear_cache(self) -> None:
        """Clear internal prediction caches."""
        self._prediction_cache.clear()
        self._diagnostics_cache = None

    @staticmethod
    def _compute_skewness(arr: np.ndarray) -> float:
        """Compute Fisher-Pearson skewness coefficient."""
        if len(arr) < 3:
            return 0.0
        mean, std = np.mean(arr), np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    @staticmethod
    def _compute_kurtosis(arr: np.ndarray) -> float:
        """Compute excess kurtosis."""
        if len(arr) < 4:
            return 0.0
        mean, std = np.mean(arr), np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3.0)

    def _compute_distances_to_train(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
    ) -> np.ndarray:
        """Compute minimum distance from each test point to training set."""
        # Efficient pairwise distance computation
        if X_train.shape[0] > 1000:
            # Use approximate nearest neighbor for large datasets
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
                nn.fit(X_train)
                distances, _ = nn.kneighbors(X_test)
                return distances.flatten()
            except ImportError:
                pass
        
        # Exact computation
        distances = np.zeros(X_test.shape[0])
        for i, x in enumerate(X_test):
            dists = np.linalg.norm(X_train - x, axis=1)
            distances[i] = np.min(dists)
        return distances


# High-level convenience functions
def quick_uncertainty_check(
    model: Any,
    X_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
) -> str:
    """
    One-line uncertainty assessment.
    
    Returns:
        Human-readable uncertainty summary
    """
    try:
        profiler = UncertaintyProfiler(model, X_train=X_train)
        diag = profiler.compute_diagnostics(X_test)
        
        status = "Well-calibrated" if diag.is_well_calibrated else "Poorly-calibrated"
        return (
            f"{status}: mean σ²={diag.mean_uncertainty:.3e}, "
            f"CV={diag.coefficient_of_variation:.2f}, "
            f"{diag.n_extrapolation_points} extrapolation points"
        )
    except Exception as e:
        return f"Uncertainty check failed: {e}"


def compare_uncertainty_profiles(
    models: Dict[str, Any],
    X_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
) -> Dict[str, UncertaintyDiagnostics]:
    """
    Compare uncertainty profiles across multiple models.
    
    Args:
        models: Dictionary mapping model names to model objects
        X_test: Test locations
        X_train: Optional training data
        
    Returns:
        Dictionary mapping model names to their diagnostics
    """
    results = {}
    for name, model in models.items():
        try:
            profiler = UncertaintyProfiler(model, X_train=X_train)
            results[name] = profiler.compute_diagnostics(X_test)
        except Exception as e:
            logger.error(f"Failed to profile {name}: {e}")
            results[name] = None
    return results