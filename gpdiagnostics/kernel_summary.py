"""
Kernel interpretation and summarization tools for GPy models.
"""

import numpy as np
from typing import Dict, Any, List, Union, Optional
import GPy

def get_kernel_structure(kern: GPy.kern.Kern) -> Union[str, List[Any]]:
    """
    Recursively parse composite kernel trees into nested structure.
    
    Args:
        kern: GPy kernel object
        
    Returns:
        Kernel name string or nested list for composite kernels
    """
    if hasattr(kern, 'parts') and kern.parts:
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
        if hasattr(param, 'values'):
            values = param.param_array
            if hasattr(values, '__iter__') and not isinstance(values, str):
                params[param.name] = [float(v) for v in values]
            else:
                params[param.name] = float(values)
    return params

def interpret_lengthscale(lengthscale: Union[float, np.ndarray]) -> str:
    """
    Interpret lengthscale magnitude with data-aware thresholds.
    
    Args:
        lengthscale: Single float or array of lengthscales
        
    Returns:
        Human-readable interpretation string
    """
    # Handle ARD lengthscales
    if isinstance(lengthscale, (list, np.ndarray)):
        ls_mean = np.mean(lengthscale)
        ls_range = f"range[{np.min(lengthscale):.2f}, {np.max(lengthscale):.2f}]"
    else:
        ls_mean = lengthscale
        ls_range = f"{ls_mean:.2f}"
    
    if ls_mean < 0.5:
        return f"Rapid variation (high frequency, {ls_range})"
    elif ls_mean > 2.0:
        return f"Smooth trends (low frequency, {ls_range})"
    else:
        return f"Moderate flexibility ({ls_range})"

def interpret_variance(variance: float, name: str = "Signal") -> str:
    """
    Interpret variance magnitude with context-aware messaging.
    
    Args:
        variance: Variance value
        name: Type of variance ("Signal" or "Noise")
        
    Returns:
        Human-readable interpretation string
    """
    if variance < 0.01:
        return f"Very low {name.lower()} (≈{variance:.3f})"
    elif variance > 10.0:
        return f"High {name.lower()} (≈{variance:.1f})"
    return f"Moderate {name.lower()} (≈{variance:.2f})"

def summarize_kernel(
    model: GPy.models.GPRegression,
    X: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive human-readable kernel interpretation.
    
    Args:
        model: Trained GPy model
        X: Training data (optional, for context-aware thresholds)
        verbose: Whether to print formatted summary
        
    Returns:
        Dictionary with structured interpretation
        
    Raises:
        ValueError: If model lacks required attributes
    """
    if not hasattr(model, 'kern'):
        raise ValueError("Model must have a 'kern' attribute")
    
    kernel = model.kern
    structure = get_kernel_structure(kernel)
    params = extract_kernel_params(kernel)
    
    # Build interpretation structure
    interpretation = {
        "kernel_structure": structure,
        "components": [],
        "composite": isinstance(structure, list)
    }
    
    # Parse each component recursively
    def parse_component(kern: GPy.kern.Kern, path: str = ""):
        if hasattr(kern, 'parts') and kern.parts:
            for i, part in enumerate(kern.parts):
                parse_component(part, f"{path}.parts[{i}]" if path else f"parts[{i}]")
        else:
            comp = {
                "type": kern.name,
                "path": path,
                "params": {},
                "interpretation": {}
            }
            
            if hasattr(kern, 'lengthscale'):
                ls = kern.lengthscale
                comp["params"]["lengthscale"] = ls.tolist() if hasattr(ls, '__iter__') else float(ls)
                comp["interpretation"]["smoothness"] = interpret_lengthscale(ls)
            
            if hasattr(kern, 'variance'):
                var = float(kern.variance)
                comp["params"]["variance"] = var
                is_noise = "White" in kern.name or "Noise" in kern.name
                comp["interpretation"]["strength"] = interpret_variance(
                    var, 
                    "Noise" if is_noise else "Signal"
                )
            
            if hasattr(kern, 'periodicity'):
                comp["params"]["periodicity"] = float(kern.periodicity)
                comp["interpretation"]["pattern"] = f"Periodic with period {kern.periodicity.values:.2f}"
            
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
        return " " * indent + f"└─ {node}"  # use the name as it comes in (do not update names)
    
    return format_node(structure)
