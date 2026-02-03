"""
Model complexity quantification for Gaussian Processes.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import GPy
import numpy as np

logger = logging.getLogger(__name__)


class ComplexityError(Exception):
    """Raised when complexity computation fails."""
    pass


@dataclass
class ComplexityThresholds:
    """
    Configurable thresholds for complexity interpretation.
    
    Log-score thresholds determine interpretation categories:
    - simple_threshold: log10(score) below this = simple model
    - complex_threshold: log10(score) above this = complex model
    """
    simple_threshold: float = 0.5
    complex_threshold: float = 1.5
    high_noise_threshold: float = 0.1
    max_reasonable_variance: float = 1e6
    min_reasonable_variance: float = 0.0
    jitter: float = 1e-10
    
    def validate(self):
        """Ensure thresholds are logically ordered."""
        if self.simple_threshold >= self.complex_threshold:
            raise ValueError(
                f"simple_threshold ({self.simple_threshold}) must be < "
                f"complex_threshold ({self.complex_threshold})"
            )
        if self.max_reasonable_variance <= self.min_reasonable_variance:
            raise ValueError("max_reasonable_variance must be > min_reasonable_variance")
        if self.jitter <= 0:
            raise ValueError("jitter must be positive")
        return self


@dataclass
class ComplexityConfig:
    """Complete configuration for complexity computation."""
    thresholds: ComplexityThresholds = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = ComplexityThresholds()


def count_kernel_components(kern: GPy.kern.Kern) -> int:
    """
    Recursively count total kernel components in composite kernels.
    """
    try:
        if not hasattr(kern, "parts"):
            return 1
            
        if not kern.parts:
            return 1
            
        if not hasattr(kern.parts, "__iter__"):
            raise ComplexityError(f"Kernel 'parts' is not iterable: {type(kern.parts)}")
            
        return sum(count_kernel_components(k) for k in kern.parts)
        
    except ComplexityError:
        raise
    except RecursionError as e:
        logger.error(f"Recursion limit hit: {e}")
        raise ComplexityError("Kernel structure too deep or circular") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ComplexityError(f"Failed to count kernel components: {e}") from e


def compute_roughness_score(kern: GPy.kern.Kern) -> float:
    """
    Compute overall function roughness as inverse lengthscale.
    """
    roughness = 0.0
    count = 0

    def traverse(k):
        nonlocal roughness, count
        
        try:
            if not hasattr(k, "parts"):
                if hasattr(k, "lengthscale"):
                    ls = k.lengthscale
                    ls_mean = np.mean(ls) if hasattr(ls, "__iter__") else ls
                    
                    if not np.isfinite(ls_mean):
                        logger.warning(f"Non-finite lengthscale: {ls_mean}")
                        return
                        
                    roughness += 1.0 / (ls_mean + 1e-10)
                    count += 1
                return
                
            if k.parts:
                for i, part in enumerate(k.parts):
                    try:
                        traverse(part)
                    except Exception as e:
                        logger.warning(f"Failed to traverse part {i}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error traversing kernel: {e}")

    try:
        traverse(kern)
    except Exception as e:
        raise ComplexityError(f"Failed to compute roughness: {e}") from e

    if count == 0:
        logger.warning("No lengthscales found, returning zero roughness")
        return 0.0
        
    return roughness / count


def compute_noise_ratio(model: GPy.models.GPRegression) -> float:
    """
    Compute signal-to-noise ratio (SNR) for the model.
    """
    try:
        if not hasattr(model, "kern"):
            raise ComplexityError("Model has no kernel")
            
        if not hasattr(model.kern, "variance"):
            logger.debug("Kernel has no variance, assuming SNR=1.0")
            return 1.0
            
        signal_var = float(model.kern.variance)
        
        if not hasattr(model, "Gaussian_noise"):
            raise ComplexityError("Model has no Gaussian_noise")
            
        if not hasattr(model.Gaussian_noise, "variance"):
            raise ComplexityError("Gaussian_noise has no variance")
            
        noise_var = float(model.Gaussian_noise.variance)
        
        if not np.isfinite(signal_var) or not np.isfinite(noise_var):
            logger.warning(f"Non-finite variances: signal={signal_var}, noise={noise_var}")
            return 1.0
            
        if noise_var < 0:
            logger.warning(f"Negative noise variance: {noise_var}")
            return 1.0
            
        return float(signal_var / (noise_var + 1e-10))
        
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"Could not compute SNR: {e}")
        return 1.0
    except ComplexityError:
        raise
    except Exception as e:
        raise ComplexityError(f"Failed to compute noise ratio: {e}") from e


def compute_complexity_score(
    model: GPy.models.GPRegression,
    X: np.ndarray,
    config: Optional[ComplexityConfig] = None
) -> Dict[str, Any]:
    """
    Comprehensive model complexity quantification.

    Args:
        model: Trained GPy model
        X: Training data for degrees of freedom calculation
        config: Complexity configuration (uses defaults if None)

    Returns:
        Dictionary with complexity score and detailed breakdown
    """
    cfg = (config or ComplexityConfig()).thresholds.validate()
    
    if X is None or not hasattr(X, "shape"):
        raise ValueError("X must be a numpy array")
        
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    
    try:
        n_components = count_kernel_components(model.kern)
        roughness = compute_roughness_score(model.kern)
        noise_ratio = compute_noise_ratio(model)

        # Effective degrees of freedom
        effective_dof = X.shape[0] * 0.5
        
        try:
            K = model.kern.K(X, X)
            noise_var = float(model.Gaussian_noise.variance)
            
            if np.isfinite(noise_var) and noise_var >= 0:
                trace_K = np.trace(K)
                if np.isfinite(trace_K) and trace_K >= 0:
                    trace_ratio = trace_K / (trace_K + noise_var * X.shape[0] + cfg.jitter)
                    effective_dof = trace_ratio * X.shape[0]
                    
        except (AttributeError, ValueError, np.linalg.LinAlgError) as e:
            logger.debug(f"Could not compute effective DOF: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in DOF computation: {e}")

        # Composite complexity score
        dof_ratio = effective_dof / X.shape[0]
        complexity_score = (
            n_components * roughness * noise_ratio / (dof_ratio + cfg.jitter)
        )

        # Interpretation using configurable thresholds
        complexity_score_log = np.log10(complexity_score + 1)
        
        if complexity_score_log < cfg.simple_threshold:
            interpretation = "Simple model (low risk of overfitting)"
            suggestions = [
                "Model is likely underfitting",
                "Consider more expressive kernel",
            ]
        elif complexity_score_log < cfg.complex_threshold:
            interpretation = "Moderate complexity (well-balanced)"
            suggestions = ["Good complexity for most applications"]
        else:
            interpretation = "High complexity (monitor for overfitting)"
            suggestions = [
                "Consider simplifying kernel",
                "Add regularization",
                "Collect more data",
            ]

        return {
            "score": float(complexity_score),
            "log_score": float(complexity_score_log),
            "interpretation": interpretation,
            "suggestions": suggestions,
            "components": {
                "n_kernel_parts": n_components,
                "roughness_score": float(roughness),
                "noise_ratio": float(noise_ratio),
                "effective_degrees_of_freedom": float(effective_dof),
            },
            "risk_factors": {
                "too_complex": complexity_score_log > cfg.complex_threshold,
                "too_simple": complexity_score_log < cfg.simple_threshold,
                "high_noise": noise_ratio < cfg.high_noise_threshold,
            },
            "config": {
                "simple_threshold": cfg.simple_threshold,
                "complex_threshold": cfg.complex_threshold,
                "high_noise_threshold": cfg.high_noise_threshold,
            }
        }
        
    except ComplexityError:
        raise
    except Exception as e:
        raise ComplexityError(f"Failed to compute complexity score: {e}") from e


def check_variance_reasonable(
    variance: float,
    config: Optional[ComplexityThresholds] = None
) -> bool:
    """
    Check if variance value is within reasonable bounds.
    
    Args:
        variance: Variance value to check
        config: Threshold configuration
        
    Returns:
        True if variance is reasonable
    """
    cfg = (config or ComplexityThresholds()).validate()
    return cfg.min_reasonable_variance < variance < cfg.max_reasonable_variance