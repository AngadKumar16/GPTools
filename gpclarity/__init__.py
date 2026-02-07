"""
gpclarity: Interpretability Toolkit for Gaussian Processes
"""

import warnings
from importlib.util import find_spec

from ._version import __version__

_GPY_AVAILABLE = find_spec("GPy") is not None

AVAILABLE = {
    "gpy": _GPY_AVAILABLE,
    "full": _GPY_AVAILABLE,
}

__all__ = [
    "__version__",
    "AVAILABLE",
    "summarize_kernel",
    "format_kernel_tree", 
    "interpret_lengthscale",
    "interpret_variance",
    "UncertaintyProfiler",
    "HyperparameterTracker",
    "compute_complexity_score",
    "count_kernel_components",
    "compute_roughness_score",
    "compute_noise_ratio",
    "DataInfluenceMap",
    "check_model_health",
    "extract_kernel_params_flat",
    "get_lengthscale",
    "get_noise_variance",
]

if _GPY_AVAILABLE:
    from .kernel_summary import (
        summarize_kernel,
        format_kernel_tree,
        interpret_lengthscale,
        extract_kernel_params_flat,
        interpret_variance,
    )
    from .uncertainty_analysis import UncertaintyProfiler
    from .hyperparam_tracker import HyperparameterTracker
    from .model_complexity import (
        compute_complexity_score,
        count_kernel_components,
        compute_roughness_score,
        compute_noise_ratio,
    )
    from .data_influence import DataInfluenceMap
    from .utils import (
        check_model_health,
        get_lengthscale,
        get_noise_variance,
    )
else:
    # Define stubs
    class _Stub:
        def __init__(self, name):
            self.name = name
        def __call__(self, *args, **kwargs):
            raise ImportError(f"{self.name} requires GPy. Install: pip install gpclarity[full]")
    
    summarize_kernel = _Stub("summarize_kernel")
    format_kernel_tree = _Stub("format_kernel_tree")
    interpret_lengthscale = _Stub("interpret_lengthscale")
    interpret_variance = _Stub("interpret_variance")
    UncertaintyProfiler = _Stub("UncertaintyProfiler")
    HyperparameterTracker = _Stub("HyperparameterTracker")
    compute_complexity_score = _Stub("compute_complexity_score")
    count_kernel_components = _Stub("count_kernel_components")
    compute_roughness_score = _Stub("compute_roughness_score")
    compute_noise_ratio = _Stub("compute_noise_ratio")
    DataInfluenceMap = _Stub("DataInfluenceMap")
    check_model_health = _Stub("check_model_health")
    extract_kernel_params_flat = _Stub("extract_kernel_params_flat")
    get_lengthscale = _Stub("get_lengthscale")
    get_noise_variance = _Stub("get_noise_variance")
    
    warnings.warn("gpclarity running in limited mode. Install with 'pip install gpclarity[full]'", ImportWarning)

__author__ = "Angad Kumar"
__email__ = "angadkumar16ak@gmail.com"