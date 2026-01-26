"""
GPDiagnostics: Interpretability Toolkit for Gaussian Processes
==============================================================

Extends GPy and emukit with human-readable diagnostics, uncertainty
analysis, and model interpretability tools. Designed for researchers
and practitioners who need trustworthy GP models.

Public API Philosophy
---------------------
This module exposes only high-level interpretability interfaces.
Internal utilities and helpers are intentionally not exposed to maintain
a clean, stable API surface for users.

Import Safety
-------------
Heavy scientific computing dependencies (GPy, emukit) are wrapped in
try/except blocks to enable lightweight installs for documentation
or analysis-only workflows.
"""

# Version source of truth - prevents circular imports in complex packages
from ._version import __version__

# Public API: High-level interpretability and diagnostic tools
# Internal utilities are intentionally not exposed here

try:
    # Core interpretability modules
    from .kernel_summary import (
        summarize_kernel,
        format_kernel_tree,
        interpret_lengthscale,
        interpret_variance,
    )
    from .uncertainty_analysis import UncertaintyProfiler
    from .hyperparam_tracker import HyperparameterTracker
    from .model_complexity import compute_complexity_score, count_kernel_components
    from .data_influence import DataInfluenceMap
    
    # Utility functions (safe to expose)
    from .utils import get_lengthscale, get_noise_variance, extract_kernel_params_flat
    
    # Flag indicating successful import of all features
    _FULL_IMPORT_SUCCESS = True
    
except ImportError as e:  # pragma: no cover
    # Graceful degradation for minimal installs
    import warnings
    
    _FULL_IMPORT_SUCCESS = False
    _IMPORT_ERROR = str(e)
    
    warnings.warn(
        f"GPDiagnostics: Some features unavailable due to missing dependency: {e}\n"
        "Install with 'pip install gpdiagnostics[full]' for complete functionality.",
        ImportWarning,
        stacklevel=2,
    )
    
    # Define stubs to prevent NameError
    summarize_kernel = None
    format_kernel_tree = None
    interpret_lengthscale = None
    interpret_variance = None
    UncertaintyProfiler = None
    HyperparameterTracker = None
    compute_complexity_score = None
    count_kernel_components = None
    DataInfluenceMap = None
    get_lengthscale = None
    get_noise_variance = None
    extract_kernel_params_flat = None


# Explicit public API declaration
__all__ = [
    # Version
    "__version__",
    
    # Kernel Summary
    "summarize_kernel",
    "format_kernel_tree",
    "interpret_lengthscale",
    "interpret_variance",
    
    # Uncertainty Analysis
    "UncertaintyProfiler",
    
    # Hyperparameter Tracking
    "HyperparameterTracker",
    
    # Model Complexity
    "compute_complexity_score",
    "count_kernel_components",
    
    # Data Influence
    "DataInfluenceMap",
    
    # Utilities
    "get_lengthscale",
    "get_noise_variance",
    "extract_kernel_params_flat",
]

# Package metadata
__author__ = "Angad Kumar"
__email__ = "angadkumar16ak@gmail.com"
__description__ = "Interpretability and Diagnostics for Gaussian Processes"
