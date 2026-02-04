"""
Model complexity quantification for Gaussian Processes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from gpclarity.exceptions import ComplexityError, KernelError
from gpclarity.kernel_summary import count_kernel_components, extract_kernel_params_flat
from gpclarity.utils import _cholesky_with_jitter, _validate_kernel_matrix

logger = logging.getLogger(__name__)


class ComplexityCategory(Enum):
    """Categorization of model complexity levels."""
    TOO_SIMPLE = auto()
    SIMPLE = auto()
    MODERATE = auto()
    COMPLEX = auto()
    TOO_COMPLEX = auto()
    
    @property
    def description(self) -> str:
        descriptions = {
            ComplexityCategory.TOO_SIMPLE: "Overly simplistic (high underfitting risk)",
            ComplexityCategory.SIMPLE: "Simple model (possible underfitting)",
            ComplexityCategory.MODERATE: "Well-balanced complexity",
            ComplexityCategory.COMPLEX: "Complex model (monitor for overfitting)",
            ComplexityCategory.TOO_COMPLEX: "Overly complex (high overfitting risk)",
        }
        return descriptions[self]
    
    @property
    def risk_level(self) -> str:
        levels = {
            ComplexityCategory.TOO_SIMPLE: "HIGH",
            ComplexityCategory.SIMPLE: "MEDIUM",
            ComplexityCategory.MODERATE: "LOW",
            ComplexityCategory.COMPLEX: "MEDIUM",
            ComplexityCategory.TOO_COMPLEX: "HIGH",
        }
        return levels[self]


@dataclass(frozen=True)
class ComplexityThresholds:
    """
    Data-adaptive thresholds for complexity interpretation.
    
    Thresholds are in log10(complexity_score) space.
    """
    too_simple: float = -0.5
    simple: float = 0.5
    complex: float = 1.5
    too_complex: float = 2.5
    high_noise_ratio: float = 0.1  # signal/noise < this is noisy
    low_signal_ratio: float = 10.0  # signal/noise > this is dominated by signal
    jitter: float = 1e-10
    
    def __post_init__(self):
        # Validate ordering
        thresholds = [self.too_simple, self.simple, self.complex, self.too_complex]
        if not all(t < u for t, u in zip(thresholds, thresholds[1:])):
            raise ValueError("Thresholds must be strictly increasing")
        if self.jitter <= 0:
            raise ValueError("jitter must be positive")
    
    def categorize(self, log_score: float) -> ComplexityCategory:
        """Categorize complexity based on log score."""
        if log_score < self.too_simple:
            return ComplexityCategory.TOO_SIMPLE
        elif log_score < self.simple:
            return ComplexityCategory.SIMPLE
        elif log_score < self.complex:
            return ComplexityCategory.MODERATE
        elif log_score < self.too_complex:
            return ComplexityCategory.COMPLEX
        return ComplexityCategory.TOO_COMPLEX


@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics for a GP model."""
    total_score: float
    log_score: float
    category: ComplexityCategory
    n_parameters: int
    n_kernel_components: int
    roughness_score: float
    signal_noise_ratio: float
    effective_degrees_of_freedom: float
    capacity_ratio: float  # DOF / n_samples
    geometric_complexity: float  # Alternative metric based on eigenvalues
    
    @property
    def is_well_specified(self) -> bool:
        """Check if model complexity is appropriate."""
        return self.category == ComplexityCategory.MODERATE
    
    @property
    def risk_factors(self) -> List[str]:
        """Identify specific risk factors."""
        risks = []
        if self.category in (ComplexityCategory.TOO_SIMPLE, ComplexityCategory.SIMPLE):
            risks.append("Underfitting risk: model may be too restrictive")
        if self.category in (ComplexityCategory.COMPLEX, ComplexityCategory.TOO_COMPLEX):
            risks.append("Overfitting risk: model may be too flexible")
        if self.signal_noise_ratio < 0.1:
            risks.append("High noise: predictions may be unreliable")
        if self.capacity_ratio > 0.8:
            risks.append("High capacity: model can memorize training data")
        return risks


class ComplexityScorer(Protocol):
    """Protocol for pluggable complexity scoring strategies."""
    def __call__(self, model: Any, X: np.ndarray, **kwargs) -> float:
        ...


@dataclass
class ComplexityAnalyzer:
    """
    Configurable complexity analysis with multiple scoring strategies.
    
    This class provides the core analysis logic, separated from
    the high-level API functions.
    """
    thresholds: ComplexityThresholds = field(default_factory=ComplexityThresholds)
    scoring_strategy: str = "default"
    
    # Registry of scoring strategies
    _strategies: Dict[str, ComplexityScorer] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        if not self._strategies:
            self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register built-in scoring strategies."""
        self._strategies["default"] = self._default_score
        self._strategies["geometric"] = self._geometric_score
        self._strategies["bayesian"] = self._bayesian_score
    
    @staticmethod
    def _default_score(model: Any, X: np.ndarray, **kwargs) -> float:
        """
        Default complexity score based on components, roughness, and SNR.
        
        Score = (n_components * roughness) / (SNR * capacity_ratio)
        """
        n_comp = kwargs.get('n_components', 1)
        roughness = kwargs.get('roughness', 1.0)
        snr = kwargs.get('snr', 1.0)
        capacity = kwargs.get('capacity_ratio', 0.5)
        jitter = kwargs.get('jitter', 1e-10)
        
        return (n_comp * roughness) / (snr * capacity + jitter)
    
    @staticmethod
    def _geometric_score(model: Any, X: np.ndarray, **kwargs) -> float:
        """
        Geometric complexity based on eigenvalue spectrum of kernel matrix.
        
        Uses effective rank as complexity measure.
        """
        try:
            K = model.kern.K(X, X)
            eigenvals = np.linalg.eigvalsh(K)
            eigenvals = np.maximum(eigenvals, 0)  # Numerical safety
            
            # Effective rank (participation ratio)
            if np.sum(eigenvals) > 0:
                effective_rank = (np.sum(eigenvals) ** 2) / (np.sum(eigenvals ** 2) + 1e-10)
                return float(effective_rank / X.shape[0])
            return 1.0
        except Exception as e:
            logger.warning(f"Geometric score failed: {e}")
            return ComplexityAnalyzer._default_score(model, X, **kwargs)
    
    @staticmethod
    def _bayesian_score(model: Any, X: np.ndarray, **kwargs) -> float:
        """
        Bayesian model complexity using log marginal likelihood curvature.
        
        Approximates complexity as trace of Fisher information.
        """
        try:
            # Approximate using gradient of log-likelihood
            if hasattr(model, 'log_likelihood') and hasattr(model, 'gradient'):
                ll = model.log_likelihood()
                grad = model.gradient
                if grad is not None and len(grad) > 0:
                    # Complexity ~ ||gradient|| / |LL| (steep LL = complex)
                    complexity = np.linalg.norm(grad) / (abs(ll) + 1.0)
                    return float(complexity)
            return ComplexityAnalyzer._default_score(model, X, **kwargs)
        except Exception as e:
            logger.warning(f"Bayesian score failed: {e}")
            return ComplexityAnalyzer._default_score(model, X, **kwargs)
    
    def analyze(self, model: Any, X: np.ndarray) -> ComplexityMetrics:
        """
        Perform comprehensive complexity analysis.
        
        Args:
            model: GP model with kern and likelihood attributes
            X: Training data (n_samples, n_features)
            
        Returns:
            ComplexityMetrics with detailed diagnostics
        """
        if X is None or not hasattr(X, 'shape'):
            raise ComplexityError("X must be a valid array")
        if X.shape[0] == 0:
            raise ComplexityError("X cannot be empty")
        
        n_samples = X.shape[0]
        
        # Collect component metrics
        n_components = self._count_components(model)
        n_params = self._count_parameters(model)
        roughness = self._compute_roughness(model)
        snr = self._compute_snr(model)
        effective_dof = self._compute_effective_dof(model, X)
        capacity_ratio = effective_dof / n_samples
        
        # Compute geometric complexity
        geom_complexity = self._strategies["geometric"](
            model, X, n_components=n_components
        )
        
        # Get scoring function
        score_fn = self._strategies.get(
            self.scoring_strategy, 
            self._default_score
        )
        
        # Compute composite score
        score = score_fn(
            model, X,
            n_components=n_components,
            roughness=roughness,
            snr=snr,
            capacity_ratio=capacity_ratio,
            jitter=self.thresholds.jitter,
        )
        
        # Categorize
        log_score = np.log10(max(score, self.thresholds.jitter))
        category = self.thresholds.categorize(log_score)
        
        return ComplexityMetrics(
            total_score=float(score),
            log_score=float(log_score),
            category=category,
            n_parameters=n_params,
            n_kernel_components=n_components,
            roughness_score=float(roughness),
            signal_noise_ratio=float(snr),
            effective_degrees_of_freedom=float(effective_dof),
            capacity_ratio=float(capacity_ratio),
            geometric_complexity=float(geom_complexity),
        )
    
    def _count_components(self, model: Any) -> int:
        """Safely count kernel components."""
        try:
            return count_kernel_components(model.kern)
        except Exception as e:
            logger.warning(f"Component counting failed: {e}")
            return 1
    
    def _count_parameters(self, model: Any) -> int:
        """Count total trainable parameters."""
        try:
            params = extract_kernel_params_flat(model)
            return len(params)
        except Exception as e:
            logger.warning(f"Parameter counting failed: {e}")
            return 0
    
    def _compute_roughness(self, model: Any) -> float:
        """Compute function roughness score."""
        try:
            return compute_roughness_score(model.kern)
        except Exception as e:
            logger.warning(f"Roughness computation failed: {e}")
            return 1.0
    
    def _compute_snr(self, model: Any) -> float:
        """Compute signal-to-noise ratio."""
        try:
            return compute_noise_ratio(model)
        except Exception as e:
            logger.warning(f"SNR computation failed: {e}")
            return 1.0
    
    def _compute_effective_dof(self, model: Any, X: np.ndarray) -> float:
        """
        Compute effective degrees of freedom using trace of hat matrix.
        
        For GP regression: DOF = trace(K @ (K + sigma^2 I)^{-1})
        """
        try:
            K = model.kern.K(X, X)
            _validate_kernel_matrix(K)
            
            noise_var = 1.0
            if hasattr(model, 'Gaussian_noise') and hasattr(model.Gaussian_noise, 'variance'):
                noise_var = float(model.Gaussian_noise.variance)
                if not np.isfinite(noise_var) or noise_var < 0:
                    noise_var = 1.0
            
            # Stable computation via eigendecomposition
            eigenvals = np.linalg.eigvalsh(K)
            eigenvals = np.maximum(eigenvals, 0)
            
            # DOF = sum(eigenvals / (eigenvals + noise_var))
            dof = np.sum(eigenvals / (eigenvals + noise_var + self.thresholds.jitter))
            
            return float(np.clip(dof, 0, X.shape[0]))
            
        except Exception as e:
            logger.debug(f"Effective DOF computation failed: {e}")
            # Fallback: use parameter count as proxy
            return float(self._count_parameters(model))


# High-level API functions
def compute_complexity_score(
    model: Any,
    X: np.ndarray,
    *,
    strategy: str = "default",
    thresholds: Optional[ComplexityThresholds] = None,
    return_diagnostics: bool = False,
) -> Union[Dict[str, Any], ComplexityMetrics]:
    """
    Comprehensive model complexity quantification.
    
    Args:
        model: Trained GP model with kern and likelihood
        X: Training data (n_samples, n_features)
        strategy: Scoring strategy ('default', 'geometric', 'bayesian')
        thresholds: Custom thresholds (uses defaults if None)
        return_diagnostics: If True, return full ComplexityMetrics object
        
    Returns:
        Dictionary summary or ComplexityMetrics object
        
    Raises:
        ComplexityError: If analysis fails
    """
    try:
        analyzer = ComplexityAnalyzer(
            thresholds=thresholds or ComplexityThresholds(),
            scoring_strategy=strategy,
        )
        
        metrics = analyzer.analyze(model, X)
        
        if return_diagnostics:
            return metrics
        
        # Build summary dictionary
        result = {
            "score": metrics.total_score,
            "log_score": metrics.log_score,
            "category": metrics.category.name,
            "interpretation": metrics.category.description,
            "risk_level": metrics.category.risk_level,
            "risk_factors": metrics.risk_factors,
            "metrics": {
                "n_parameters": metrics.n_parameters,
                "n_kernel_components": metrics.n_kernel_components,
                "roughness_score": metrics.roughness_score,
                "signal_noise_ratio": metrics.signal_noise_ratio,
                "effective_dof": metrics.effective_degrees_of_freedom,
                "capacity_ratio": metrics.capacity_ratio,
            },
            "recommendations": _generate_recommendations(metrics),
        }
        
        return result
        
    except ComplexityError:
        raise
    except Exception as e:
        raise ComplexityError(f"Complexity analysis failed: {e}") from e


def compute_roughness_score(kern: Any) -> float:
    """
    Compute function roughness as inverse of characteristic lengthscale.
    
    Higher roughness = more wiggly function = higher complexity.
    """
    roughness_values = []
    
    def traverse(kernel: Any, path: str = "") -> None:
        """Recursively collect lengthscale-based roughness."""
        kernel_name = getattr(kernel, 'name', 'unknown')
        current_path = f"{path}.{kernel_name}" if path else kernel_name
        
        # Check for lengthscale attribute
        if hasattr(kernel, 'lengthscale'):
            try:
                ls = kernel.lengthscale
                if hasattr(ls, 'values'):
                    ls_val = ls.values
                elif hasattr(ls, 'param_array'):
                    ls_val = ls.param_array
                else:
                    ls_val = ls
                
                arr = np.atleast_1d(ls_val)
                
                # Use harmonic mean for ARD (penalizes small lengthscales more)
                if len(arr) > 1:
                    # Filter out invalid values
                    valid = arr[np.isfinite(arr) & (arr > 0)]
                    if len(valid) > 0:
                        hmean = len(valid) / np.sum(1.0 / valid)
                        roughness_values.append(1.0 / hmean)
                else:
                    ls_float = float(arr[0])
                    if np.isfinite(ls_float) and ls_float > 0:
                        roughness_values.append(1.0 / ls_float)
                        
            except Exception as e:
                logger.debug(f"Could not extract lengthscale from {current_path}: {e}")
        
        # Recurse into composite kernels
        if hasattr(kernel, 'parts') and kernel.parts:
            for i, part in enumerate(kernel.parts):
                traverse(part, f"{current_path}[{i}]")
    
    try:
        traverse(kern)
    except RecursionError:
        raise ComplexityError("Kernel structure too deep (possible circular reference)")
    except Exception as e:
        raise ComplexityError(f"Roughness computation failed: {e}") from e
    
    if not roughness_values:
        logger.debug("No lengthscales found, returning unit roughness")
        return 1.0
    
    # Return geometric mean of roughness values
    log_roughness = np.mean(np.log(roughness_values))
    return float(np.exp(log_roughness))


def compute_noise_ratio(model: Any) -> float:
    """
    Compute signal-to-noise ratio (variance_signal / variance_noise).
    
    Returns:
        SNR value (>1 means signal dominates, <1 means noise dominates)
    """
    try:
        # Extract signal variance from kernel
        signal_var = _extract_signal_variance(model)
        
        # Extract noise variance from likelihood
        noise_var = _extract_noise_variance(model)
        
        if not np.isfinite(signal_var) or not np.isfinite(noise_var):
            logger.warning("Non-finite variance values detected")
            return 1.0
        
        if noise_var <= 0:
            logger.warning("Zero or negative noise variance")
            return 10.0  # Assume high SNR if no noise
        
        snr = signal_var / noise_var
        
        # Sanity bounds
        return float(np.clip(snr, 1e-6, 1e6))
        
    except Exception as e:
        logger.debug(f"SNR computation failed: {e}")
        return 1.0


def _extract_signal_variance(model: Any) -> float:
    """Extract signal variance from model kernel."""
    if not hasattr(model, 'kern'):
        raise ComplexityError("Model has no kernel")
    
    kern = model.kern
    
    # Try to get variance from kernel
    if hasattr(kern, 'variance'):
        try:
            return float(kern.variance)
        except (TypeError, ValueError):
            pass
    
    # For composite kernels, sum variances
    if hasattr(kern, 'parts') and kern.parts:
        total_var = 0.0
        for part in kern.parts:
            if hasattr(part, 'variance'):
                try:
                    total_var += float(part.variance)
                except (TypeError, ValueError):
                    pass
        if total_var > 0:
            return total_var
    
    # Fallback: estimate from kernel diagonal
    try:
        # Sample variance from kernel matrix diagonal
        x_dummy = np.zeros((10, 1))  # Dummy input
        K = kern.K(x_dummy, x_dummy)
        return float(np.mean(np.diag(K)))
    except Exception:
        pass
    
    logger.debug("Could not extract signal variance, using default")
    return 1.0


def _extract_noise_variance(model: Any) -> float:
    """Extract noise variance from model likelihood."""
    # Try Gaussian_noise attribute (GPy style)
    if hasattr(model, 'Gaussian_noise'):
        if hasattr(model.Gaussian_noise, 'variance'):
            try:
                return float(model.Gaussian_noise.variance)
            except (TypeError, ValueError):
                pass
    
    # Try likelihood attribute (general)
    if hasattr(model, 'likelihood'):
        lik = model.likelihood
        if hasattr(lik, 'variance'):
            try:
                return float(lik.variance)
            except (TypeError, ValueError):
                pass
    
    logger.debug("Could not extract noise variance, using default")
    return 0.1


def _generate_recommendations(metrics: ComplexityMetrics) -> List[str]:
    """Generate actionable recommendations based on metrics."""
    recs = []
    
    if metrics.category == ComplexityCategory.TOO_SIMPLE:
        recs.extend([
            "Add more kernel components (e.g., RBF + Linear)",
            "Increase kernel flexibility (reduce lengthscale)",
            "Check if model captures all data trends",
        ])
    elif metrics.category == ComplexityCategory.SIMPLE:
        recs.extend([
            "Consider more expressive kernel structure",
            "Verify that lengthscales are appropriate for data",
        ])
    elif metrics.category == ComplexityCategory.COMPLEX:
        recs.extend([
            "Monitor validation performance for overfitting",
            "Consider kernel simplification or regularization",
            "Collect more training data if possible",
        ])
    elif metrics.category == ComplexityCategory.TOO_COMPLEX:
        recs.extend([
            "Simplify kernel structure (remove components)",
            "Add strong priors on hyperparameters",
            "Increase noise variance to regularize",
            "Use sparse approximation methods",
        ])
    
    # SNR-specific recommendations
    if metrics.signal_noise_ratio < 0.1:
        recs.append("High noise level: consider denoising preprocessing")
    elif metrics.signal_noise_ratio > 100:
        recs.append("Very clean signal: can use simpler model")
    
    # Capacity recommendations
    if metrics.capacity_ratio > 0.9:
        recs.append("Model has capacity to interpolate: risk of overfitting")
    
    return recs


# Backwards compatibility wrappers
def check_variance_reasonable(variance: float, max_val: float = 1e6, min_val: float = 0.0) -> bool:
    """Check if variance is within reasonable bounds."""
    return min_val < variance < max_val


# Convenience function for quick assessment
def quick_complexity_check(model: Any, X: np.ndarray) -> str:
    """
    One-line complexity assessment.
    
    Returns:
        Human-readable complexity assessment string
    """
    try:
        result = compute_complexity_score(model, X)
        cat = result.get('category', 'UNKNOWN')
        interp = result.get('interpretation', '')
        score = result.get('log_score', 0)
        return f"{cat}: {interp} (log-score={score:.2f})"
    except Exception as e:
        return f"Could not assess complexity: {e}"