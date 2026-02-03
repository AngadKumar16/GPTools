"""
Kernel interpretation and summarization tools for GPy models.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import GPy
import numpy as np


@dataclass
class LengthscaleThresholds:
    """Configurable thresholds for lengthscale interpretation."""
    rapid_variation: float = 0.5
    smooth_trend: float = 2.0
    
    def validate(self):
        """Ensure thresholds are logically ordered."""
        if not 0 < self.rapid_variation < self.smooth_trend:
            raise ValueError(
                f"Thresholds must satisfy 0 < rapid_variation ({self.rapid_variation}) "
                f"< smooth_trend ({self.smooth_trend})"
            )
        return self


@dataclass
class VarianceThresholds:
    """Configurable thresholds for variance interpretation."""
    very_low: float = 0.01
    high: float = 10.0
    
    def validate(self):
        """Ensure thresholds are positive."""
        if self.very_low <= 0 or self.high <= 0:
            raise ValueError("Variance thresholds must be positive")
        if self.very_low >= self.high:
            raise ValueError(f"very_low ({self.very_low}) must be < high ({self.high})")
        return self


@dataclass
class InterpretationConfig:
    """Complete configuration for kernel interpretation."""
    lengthscale: LengthscaleThresholds = None
    variance: VarianceThresholds = None
    
    def __post_init__(self):
        if self.lengthscale is None:
            self.lengthscale = LengthscaleThresholds()
        if self.variance is None:
            self.variance = VarianceThresholds()
        self.lengthscale.validate()
        self.variance.validate()


def get_kernel_structure(kern: GPy.kern.Kern) -> Union[str, List[Any]]:
    """
    Recursively parse composite kernel trees into nested structure.

    Args:
        kern: GPy kernel object

    Returns:
        Kernel name string or nested list for composite kernels
    """
    if hasattr(kern, "parts") and kern.parts:
        return [get_kernel_structure(k) for k in kern.parts]
    return kern.name


def extract_kernel_params(kern: GPy.kern.Kern) -> Dict[str, float]:
    """
    Extract all hyperparameters with proper handling of constraints and transformations.

    Args:
        kern: GPy kernel object

    Returns:
        Dictionary mapping parameter names to values
    """
    params = {}
    for param in kern.parameters:
        # Get raw value before transformation
        if hasattr(param, "values"):
            values = param.param_array
            if hasattr(values, "__iter__") and not isinstance(values, str):
                params[param.name] = [float(v) for v in values]
            else:
                params[param.name] = float(values)
    return params


def interpret_lengthscale(
    lengthscale: Union[float, np.ndarray],
    config: Optional[LengthscaleThresholds] = None
) -> str:
    """
    Interpret lengthscale magnitude with data-aware thresholds.

    Args:
        lengthscale: Single float or array of lengthscales
        config: Threshold configuration (uses defaults if None)

    Returns:
        Human-readable interpretation string
    """
    cfg = (config or LengthscaleThresholds()).validate()
    
    # Handle ARD lengthscales
    if isinstance(lengthscale, (list, np.ndarray)):
        ls_mean = np.mean(lengthscale)
        ls_range = f"range[{np.min(lengthscale):.2f}, {np.max(lengthscale):.2f}]"
    else:
        ls_mean = lengthscale
        ls_range = f"{ls_mean:.2f}"

    if ls_mean < cfg.rapid_variation:
        return f"Rapid variation (high frequency, {ls_range})"
    elif ls_mean > cfg.smooth_trend:
        return f"Smooth trends (low frequency, {ls_range})"
    else:
        return f"Moderate flexibility ({ls_range})"


def interpret_variance(
    variance: float,
    name: str = "Signal",
    config: Optional[VarianceThresholds] = None
) -> str:
    """
    Interpret variance magnitude with context-aware messaging.

    Args:
        variance: Variance value
        name: Type of variance ("Signal" or "Noise")
        config: Threshold configuration (uses defaults if None)

    Returns:
        Human-readable interpretation string
    """
    cfg = (config or VarianceThresholds()).validate()
    
    if variance < cfg.very_low:
        return f"Very low {name.lower()} (≈{variance:.3f})"
    elif variance > cfg.high:
        return f"High {name.lower()} (≈{variance:.1f})"
    return f"Moderate {name.lower()} (≈{variance:.2f})"


def summarize_kernel(
    model: GPy.models.GPRegression,
    X: Optional[np.ndarray] = None,
    verbose: bool = True,
    config: Optional[InterpretationConfig] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive human-readable kernel interpretation.

    Args:
        model: Trained GPy model
        X: Training data (optional, for context-aware thresholds)
        verbose: Whether to print formatted summary
        config: Interpretation configuration (uses defaults if None)

    Returns:
        Dictionary with structured interpretation

    Raises:
        ValueError: If model lacks required attributes
    """
    if not hasattr(model, "kern"):
        raise ValueError("Model must have a 'kern' attribute")

    cfg = config or InterpretationConfig()
    kernel = model.kern
    structure = get_kernel_structure(kernel)
    params = extract_kernel_params(kernel)

    # Build interpretation structure
    interpretation = {
        "kernel_structure": structure,
        "components": [],
        "composite": isinstance(structure, list),
        "config": {
            "lengthscale_thresholds": {
                "rapid_variation": cfg.lengthscale.rapid_variation,
                "smooth_trend": cfg.lengthscale.smooth_trend,
            },
            "variance_thresholds": {
                "very_low": cfg.variance.very_low,
                "high": cfg.variance.high,
            }
        }
    }

    # Parse each component recursively
    def parse_component(kern: GPy.kern.Kern, path: str = ""):
        if hasattr(kern, "parts") and kern.parts:
            for i, part in enumerate(kern.parts):
                parse_component(part, f"{path}.parts[{i}]" if path else f"parts[{i}]")
        else:
            comp = {"type": kern.name, "path": path, "params": {}, "interpretation": {}}

            if hasattr(kern, "lengthscale"):
                ls = kern.lengthscale
                comp["params"]["lengthscale"] = (
                    ls.tolist() if hasattr(ls, "__iter__") else float(ls)
                )
                comp["interpretation"]["smoothness"] = interpret_lengthscale(ls, cfg.lengthscale)

            if hasattr(kern, "variance"):
                var = float(kern.variance)
                comp["params"]["variance"] = var
                is_noise = "White" in kern.name or "Noise" in kern.name
                comp["interpretation"]["strength"] = interpret_variance(
                    var, "Noise" if is_noise else "Signal", cfg.variance
                )

            if hasattr(kern, "periodicity"):
                comp["params"]["periodicity"] = float(kern.periodicity)
                comp["interpretation"][
                    "pattern"
                ] = f"Periodic with period {kern.periodicity.values:.2f}"

            interpretation["components"].append(comp)

    parse_component(kernel)

    # Overall assessment
    if interpretation["composite"]:
        n_components = len(interpretation["components"])
        interpretation["overall"] = f"Composite kernel with {n_components} components"
    else:
        interpretation["overall"] = "Single kernel"

    # Print formatted summary if requested
    if verbose:
        _print_kernel_summary(interpretation)

    return interpretation


def _print_kernel_summary(interpretation: Dict[str, Any]):
    """Pretty print kernel summary."""
    print("\n KERNEL SUMMARY")
    print("=" * 50)
    print(f"Structure: {interpretation['kernel_structure']}")
    
    # Print config info
    cfg = interpretation.get("config", {})
    if cfg:
        print(f"\nThresholds:")
        ls_cfg = cfg.get("lengthscale_thresholds", {})
        print(f"  Lengthscale: rapid<{ls_cfg.get('rapid_variation', 0.5)}, "
              f"smooth>{ls_cfg.get('smooth_trend', 2.0)}")
        var_cfg = cfg.get("variance_thresholds", {})
        print(f"  Variance: very_low<{var_cfg.get('very_low', 0.01)}, "
              f"high>{var_cfg.get('high', 10.0)}")
    print()

    for comp in interpretation["components"]:
        print(f" {comp['type']} ({comp['path']})")
        for key, val in comp["params"].items():
            print(f"  └─ {key}: {val}")
        for key, val in comp["interpretation"].items():
            print(f"    {val}")
        print()


def format_kernel_tree(model: GPy.models.GPRegression) -> str:
    """
    Pretty-print kernel tree structure using the original kernel names.
    """
    structure = get_kernel_structure(model.kern)

    def format_node(node, indent=0):
        if isinstance(node, list):
            return "\n".join(format_node(n, indent + 2) for n in node)
        return (
            " " * indent + f"└─ {node}"
        )  # use the name as it comes in (do not update names)

    return format_node(structure)