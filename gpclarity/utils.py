"""
Model complexity quantification for Gaussian Processes.
"""

import logging
from typing import Any, Dict

import GPy
import numpy as np

logger = logging.getLogger(__name__)


class ComplexityError(Exception):
    """Raised when complexity computation fails."""
    pass


def count_kernel_components(kern: GPy.kern.Kern) -> int:
    """
    Recursively count total kernel components in composite kernels.

    Args:
        kern: GPy kernel object

    Returns:
        Total number of kernel components

    Raises:
        ComplexityError: If kernel traversal fails unexpectedly
    """
    try:
        if not hasattr(kern, "parts"):
            return 1
            
        if not kern.parts:
            return 1
            
        # Validate parts is iterable
        if not hasattr(kern.parts, "__iter__"):
            raise ComplexityError(f"Kernel 'parts' is not iterable: {type(kern.parts)}")
            
        return sum(count_kernel_components(k) for k in kern.parts)
        
    except ComplexityError:
        raise
    except RecursionError as e:
        logger.error(f"Recursion limit hit in kernel counting (circular reference?): {e}")
        raise ComplexityError("Kernel structure too deep or circular") from e
    except Exception as e:
        logger.error(f"Unexpected error counting kernel components: {e}")
        raise ComplexityError(f"Failed to count kernel components: {e}") from e


def compute_roughness_score(kern: GPy.kern.Kern) -> float:
    """
    Compute overall function roughness as inverse lengthscale.

    Args:
        kern: GPy kernel object

    Returns:
        Roughness score (higher = more wiggly)

    Raises:
        ComplexityError: If roughness computation fails
    """
    roughness = 0.0
    count = 0

    def traverse(k):
        nonlocal roughness, count
        
        try:
            if not hasattr(k, "parts"):
                # Leaf kernel
                if hasattr(k, "lengthscale"):
                    ls = k.lengthscale
                    ls_mean = np.mean(ls) if hasattr(ls, "__iter__") else ls
                    
                    if not np.isfinite(ls_mean):
                        logger.warning(f"Non-finite lengthscale encountered: {ls_mean}")
                        return
                        
                    roughness += 1.0 / (ls_mean + 1e-10)
                    count += 1
                return
                
            if k.parts:
                for i, part in enumerate(k.parts):
                    try:
                        traverse(part)
                    except Exception as e:
                        logger.warning(f"Failed to traverse kernel part {i}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error traversing kernel: {e}")

    try:
        traverse(kern)
    except Exception as e:
        logger.error(f"Roughness score computation failed: {e}")
        raise ComplexityError(f"Failed to compute roughness: {e}") from e

    if count == 0:
        logger.warning("No lengthscales found in kernel, returning zero roughness")
        return 0.0
        
    return roughness / count


def compute_noise_ratio(model: GPy.models.GPRegression) -> float:
    """
    Compute signal-to-noise ratio (SNR) for the model.

    Args:
        model: Trained GPy model

    Returns:
        SNR = signal_variance / noise_variance (returns 1.0 if indeterminate)

    Raises:
        ComplexityError: If SNR computation fails unexpectedly
    """
    try:
        if not hasattr(model, "kern"):
            raise ComplexityError("Model has no kernel")
            
        if not hasattr(model.kern, "variance"):
            # Some kernels don't have variance (e.g., combination kernels)
            logger.debug("Kernel has no variance attribute, assuming SNR=1.0")
            return 1.0
            
        signal_var = float(model.kern.variance)
        
        if not hasattr(model, "Gaussian_noise"):
            raise ComplexityError("Model has no Gaussian_noise attribute")
            
        if not hasattr(model.Gaussian_noise, "variance"):
            raise ComplexityError("Gaussian_noise has no variance attribute")
            
        noise_var = float(model.Gaussian_noise.variance)
        
        # Handle edge cases
        if not np.isfinite(signal_var) or not np.isfinite(noise_var):
            logger.warning(f"Non-finite variance values: signal={signal_var}, noise={noise_var}")
            return 1.0
            
        if noise_var < 0:
            logger.warning(f"Negative noise variance: {noise_var}")
            return 1.0
            
        return float(signal_var / (noise_var + 1e-10))
        
    except (AttributeError, TypeError, ValueError) as e:
        # Expected failures for non-standard model structures
        logger.debug(f"Could not compute noise ratio (expected for some models): {e}")
        return 1.0
    except ComplexityError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error computing noise ratio: {e}")
        raise ComplexityError(f"Failed to compute noise ratio: {e}") from e


def compute_complexity_score(
    model: GPy.models.GPRegression, X: np.ndarray
) -> Dict[str, Any]:
    """
    Comprehensive model complexity quantification.

    Combines multiple metrics: kernel components, roughness, noise ratio,
    and effective degrees of freedom.

    Args:
        model: Trained GPy model
        X: Training data for degrees of freedom calculation

    Returns:
        Dictionary with complexity score and detailed breakdown

    Raises:
        ComplexityError: If computation fails
        ValueError: If inputs are invalid
    """
    if X is None or not hasattr(X, "shape"):
        raise ValueError("X must be a numpy array")
        
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    
    try:
        n_components = count_kernel_components(model.kern)
        roughness = compute_roughness_score(model.kern)
        noise_ratio = compute_noise_ratio(model)

        # Effective degrees of freedom (approximation)
        effective_dof = X.shape[0] * 0.5  # default fallback
        
        try:
            K = model.kern.K(X, X)
            noise_var = float(model.Gaussian_noise.variance)
            
            if not np.isfinite(noise_var):
                logger.warning(f"Non-finite noise variance: {noise_var}")
            else:
                trace_K = np.trace(K)
                if np.isfinite(trace_K) and trace_K >= 0:
                    trace_ratio = trace_K / (trace_K + noise_var * X.shape[0] + 1e-10)
                    effective_dof = trace_ratio * X.shape[0]
                    
        except (AttributeError, ValueError, np.linalg.LinAlgError) as e:
            logger.debug(f"Could not compute effective DOF: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in effective DOF computation: {e}")

        # Composite complexity score (0 = simple, ∞ = complex)
        dof_ratio = effective_dof / X.shape[0]
        complexity_score = (
            n_components * roughness * noise_ratio / (dof_ratio + 1e-10)
        )

        # Interpretation thresholds (adaptive)
        complexity_score_log = np.log10(complexity_score + 1)
        
        if complexity_score_log < 0.5:
            interpretation = "Simple model (low risk of overfitting)"
            suggestions = [
                "Model is likely underfitting",
                "Consider more expressive kernel",
            ]
        elif complexity_score_log < 1.5:
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
                "too_complex": complexity_score_log > 1.5,
                "too_simple": complexity_score_log < 0.5,
                "high_noise": noise_ratio < 0.1,
            },
        }
        
    except ComplexityError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in complexity score computation: {e}")
        raise ComplexityError(f"Failed to compute complexity score: {e}") from e