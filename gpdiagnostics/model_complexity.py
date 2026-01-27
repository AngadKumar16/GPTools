"""
Model complexity quantification for Gaussian Processes.
"""

import numpy as np
from typing import Dict, Any
import GPy

def count_kernel_components(kern: GPy.kern.Kern) -> int:
    """
    Recursively count total kernel components in composite kernels.
    
    Args:
        kern: GPy kernel object
        
    Returns:
        Total number of kernel components
    """
    if hasattr(kern, 'parts') and kern.parts:
        return sum(count_kernel_components(k) for k in kern.parts)
    return 1

def compute_roughness_score(kern: GPy.kern.Kern) -> float:
    """
    Compute overall function roughness as inverse lengthscale.
    
    Args:
        kern: GPy kernel object
        
    Returns:
        Roughness score (higher = more wiggly)
    """
    roughness = 0.0
    count = 0
    
    def traverse(k):
        nonlocal roughness, count
        if hasattr(k, 'parts') and k.parts:
            for part in k.parts:
                traverse(part)
        elif hasattr(k, 'lengthscale'):
            ls = k.lengthscale
            ls_mean = np.mean(ls) if hasattr(ls, '__iter__') else ls
            roughness += 1.0 / (ls_mean + 1e-10)
            count += 1
    
    traverse(kern)
    return roughness / max(count, 1)

def compute_noise_ratio(model: GPy.models.GPRegression) -> float:
    """
    Compute signal-to-noise ratio (SNR) for the model.
    
    Args:
        model: Trained GPy model
        
    Returns:
        SNR = signal_variance / noise_variance
    """
    try:
        signal_var = model.kern.variance.values
        noise_var = model.likelihood.variance.values
        return float(signal_var / (noise_var + 1e-10))
    except:
        return 1.0

def compute_complexity_score(
    model: GPy.models.GPRegression, 
    X: np.ndarray
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
    """
    n_components = count_kernel_components(model.kern)
    roughness = compute_roughness_score(model.kern)
    noise_ratio = compute_noise_ratio(model)
    
    # Effective degrees of freedom (approximation)
    try:
        K = model.kern.K(X, X)
        noise_var = np.exp(model.likelihood.log_variance)
        trace_K = np.trace(K)
        trace_ratio = trace_K / (trace_K + noise_var * X.shape[0])
        effective_dof = trace_ratio * X.shape[0]
    except:
        effective_dof = X.shape[0] * 0.5
    
    # Composite complexity score (0 = simple, ∞ = complex)
    complexity_score = n_components * roughness * noise_ratio / (effective_dof / X.shape[0] + 1e-10)
    
    # Interpretation thresholds (adaptive)
    complexity_score_log = np.log10(complexity_score + 1)
    if complexity_score_log < 0.5:
        interpretation = "Simple model (low risk of overfitting)"
        suggestions = ["Model is likely underfitting", "Consider more expressive kernel"]
    elif complexity_score_log < 1.5:
        interpretation = "Moderate complexity (well-balanced)"
        suggestions = ["Good complexity for most applications"]
    else:
        interpretation = "High complexity (monitor for overfitting)"
        suggestions = ["Consider simplifying kernel", "Add regularization", "Collect more data"]
    
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
        }
    }
