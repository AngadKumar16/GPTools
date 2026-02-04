"""
Kernel interpretation and summarization tools for GPy models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Union

import numpy as np

from gpclarity.exceptions import KernelError

logger = logging.getLogger(__name__)


class SmoothnessCategory(Enum):
    """Categorization of lengthscale interpretations."""
    RAPID_VARIATION = auto()
    MODERATE = auto()
    SMOOTH_TREND = auto()
    
    def describe(self) -> str:
        descriptions = {
            SmoothnessCategory.RAPID_VARIATION: "Rapid variation (high frequency)",
            SmoothnessCategory.MODERATE: "Moderate flexibility",
            SmoothnessCategory.SMOOTH_TREND: "Smooth trends (low frequency)",
        }
        return descriptions[self]


class VarianceCategory(Enum):
    """Categorization of variance interpretations."""
    VERY_LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    
    def describe(self, name: str = "signal") -> str:
        return f"{self.name.replace('_', ' ').title()} {name.lower()}"


@dataclass(frozen=True)
class LengthscaleThresholds:
    """Configurable thresholds for lengthscale interpretation."""
    rapid_variation: float = 0.5
    smooth_trend: float = 2.0
    
    def __post_init__(self):
        if not 0 < self.rapid_variation < self.smooth_trend:
            raise ValueError(
                f"Thresholds must satisfy 0 < rapid_variation ({self.rapid_variation}) "
                f"< smooth_trend ({self.smooth_trend})"
            )
    
    def categorize(self, lengthscale: float) -> SmoothnessCategory:
        """Categorize a lengthscale value."""
        if lengthscale < self.rapid_variation:
            return SmoothnessCategory.RAPID_VARIATION
        elif lengthscale > self.smooth_trend:
            return SmoothnessCategory.SMOOTH_TREND
        return SmoothnessCategory.MODERATE


@dataclass(frozen=True)
class VarianceThresholds:
    """Configurable thresholds for variance interpretation."""
    very_low: float = 0.01
    high: float = 10.0
    
    def __post_init__(self):
        if self.very_low <= 0 or self.high <= 0:
            raise ValueError("Variance thresholds must be positive")
        if self.very_low >= self.high:
            raise ValueError(
                f"very_low ({self.very_low}) must be < high ({self.high})"
            )
    
    def categorize(self, variance: float) -> VarianceCategory:
        """Categorize a variance value."""
        if variance < self.very_low:
            return VarianceCategory.VERY_LOW
        elif variance > self.high:
            return VarianceCategory.HIGH
        return VarianceCategory.MODERATE


@dataclass
class InterpretationConfig:
    """Complete configuration for kernel interpretation."""
    lengthscale: LengthscaleThresholds = field(
        default_factory=LengthscaleThresholds
    )
    variance: VarianceThresholds = field(default_factory=VarianceThresholds)
    
    def __post_init__(self):
        # Ensure proper types
        if isinstance(self.lengthscale, dict):
            self.lengthscale = LengthscaleThresholds(**self.lengthscale)
        if isinstance(self.variance, dict):
            self.variance = VarianceThresholds(**self.variance)


@dataclass
class KernelComponent:
    """Represents a single kernel component with interpretation."""
    name: str
    kernel_type: str
    path: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    interpretations: Dict[str, str] = field(default_factory=dict)
    is_composite: bool = False
    children: List["KernelComponent"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "type": self.kernel_type,
            "parameters": self.parameters,
            "interpretations": self.interpretations,
        }
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


class KernelVisitor(Protocol):
    """Protocol for kernel tree visitors."""
    def visit(self, kernel: Any, path: str = "") -> Optional[KernelComponent]:
        ...


class KernelInterpreter:
    """
    Interpretable kernel analysis with pluggable strategies.
    
    This class provides the core interpretation logic, separated from
    the high-level API functions.
    """
    
    # Registry of kernel-specific interpreters
    _interpreters: Dict[str, Callable[[Any, InterpretationConfig], Dict[str, Any]]] = {}
    
    def __init__(self, config: Optional[InterpretationConfig] = None):
        self.config = config or InterpretationConfig()
    
    @classmethod
    def register_kernel(cls, kernel_type: str):
        """Decorator to register interpreter for specific kernel type."""
        def decorator(func: Callable[[Any, InterpretationConfig], Dict[str, Any]]):
            cls._interpreters[kernel_type] = func
            return func
        return decorator
    
    def interpret(self, kernel: Any, path: str = "") -> KernelComponent:
        """
        Interpret a kernel or kernel component.
        
        Args:
            kernel: GPy kernel object
            path: Hierarchical path string
            
        Returns:
            KernelComponent with interpretations
        """
        kernel_name = getattr(kernel, 'name', 'unknown')
        kernel_type = type(kernel).__name__
        
        # Check for registered handler
        if kernel_type in self._interpreters:
            params = self._interpreters[kernel_type](kernel, self.config)
        else:
            params = self._interpret_generic(kernel)
        
        # Build component
        component = KernelComponent(
            name=kernel_name,
            kernel_type=kernel_type,
            path=path,
            parameters=params.get("parameters", {}),
            interpretations=params.get("interpretations", {}),
            is_composite=params.get("is_composite", False),
        )
        
        # Handle composite kernels recursively
        if component.is_composite and hasattr(kernel, "parts"):
            for i, part in enumerate(kernel.parts):
                child_path = f"{path}.parts[{i}]" if path else f"parts[{i}]"
                child = self.interpret(part, child_path)
                component.children.append(child)
        
        return component
    
    def _interpret_generic(self, kernel: Any) -> Dict[str, Any]:
        """Generic interpretation fallback."""
        result = {"parameters": {}, "interpretations": {}, "is_composite": False}
        
        # Extract common parameters
        if hasattr(kernel, "lengthscale"):
            ls = kernel.lengthscale
            ls_values = self._extract_values(ls)
            result["parameters"]["lengthscale"] = ls_values
            
            # Interpret ARD lengthscales
            if isinstance(ls_values, (list, np.ndarray)) and len(ls_values) > 1:
                mean_ls = float(np.mean(ls_values))
                range_str = f"[{np.min(ls_values):.2f}, {np.max(ls_values):.2f}]"
                category = self.config.lengthscale.categorize(mean_ls)
                result["interpretations"]["smoothness"] = (
                    f"{category.describe()} (ARD, range: {range_str})"
                )
            else:
                ls_float = float(ls_values[0]) if isinstance(ls_values, list) else float(ls_values)
                category = self.config.lengthscale.categorize(ls_float)
                result["interpretations"]["smoothness"] = category.describe()
        
        if hasattr(kernel, "variance"):
            var = float(kernel.variance)
            result["parameters"]["variance"] = var
            category = self.config.variance.categorize(var)
            name = "Noise" if self._is_noise_kernel(kernel) else "Signal"
            result["interpretations"]["strength"] = category.describe(name)
        
        if hasattr(kernel, "periodicity"):
            per = float(kernel.periodicity)
            result["parameters"]["periodicity"] = per
            result["interpretations"]["pattern"] = f"Periodic (period={per:.2f})"
        
        # Check if composite
        if hasattr(kernel, "parts") and kernel.parts:
            result["is_composite"] = True
        
        return result
    
    @staticmethod
    def _extract_values(param: Any) -> Union[float, List[float], np.ndarray]:
        """Safely extract values from GPy parameter."""
        if param is None:
            return 0.0
        
        if hasattr(param, "values"):
            val = param.values
        elif hasattr(param, "param_array"):
            val = param.param_array
        else:
            val = param
        
        arr = np.atleast_1d(val)
        if len(arr) == 1:
            return float(arr[0])
        return arr.tolist() if isinstance(arr, np.ndarray) else list(arr)
    
    @staticmethod
    def _is_noise_kernel(kernel: Any) -> bool:
        """Determine if kernel represents noise."""
        name = getattr(kernel, 'name', '').lower()
        return any(n in name for n in ['white', 'noise', 'bias'])


class KernelSummaryFormatter:
    """Formats kernel summaries for different output formats."""
    
    def __init__(self, component: KernelComponent):
        self.root = component
    
    def to_text(self, verbose: bool = True) -> str:
        """Generate human-readable text summary."""
        lines = [
            "\n╔" + "═" * 58 + "╗",
            "║" + " KERNEL SUMMARY".center(58) + "║",
            "╚" + "═" * 58 + "╝\n",
        ]
        
        # Configuration
        lines.append("Configuration:")
        lines.append(f"  Lengthscale: rapid<{self.root.interpretations.get('lengthscale_rapid', 0.5)}, "
                    f"smooth>{self.root.interpretations.get('lengthscale_smooth', 2.0)}")
        lines.append(f"  Variance: very_low<{self.root.interpretations.get('variance_low', 0.01)}, "
                    f"high>{self.root.interpretations.get('variance_high', 10.0)}\n")
        
        # Tree structure
        lines.append("Structure:")
        lines.append(self._format_tree(self.root))
        lines.append("")
        
        # Detailed components
        lines.append("Components:")
        lines.append(self._format_components(self.root))
        
        return "\n".join(lines)
    
    def _format_tree(self, component: KernelComponent, prefix: str = "", is_last: bool = True) -> str:
        """Format tree structure with box-drawing characters."""
        connector = "└── " if is_last else "├── "
        line = prefix + connector + component.kernel_type
        if component.name != component.kernel_type:
            line += f" ({component.name})"
        
        lines = [line]
        
        if component.children:
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(component.children):
                is_last_child = (i == len(component.children) - 1)
                lines.append(self._format_tree(child, new_prefix, is_last_child))
        
        return "\n".join(lines)
    
    def _format_components(self, component: KernelComponent, depth: int = 0) -> str:
        """Format component details."""
        lines = []
        indent = "  " * depth
        
        if not component.is_composite or depth == 0:
            lines.append(f"{indent}【{component.kernel_type}】 {component.path}")
            for key, val in component.parameters.items():
                lines.append(f"{indent}  ├─ {key}: {val}")
            for key, val in component.interpretations.items():
                lines.append(f"{indent}  └─ {val}")
            lines.append("")
        
        for child in component.children:
            lines.append(self._format_components(child, depth + 1))
        
        return "\n".join(lines)
    
    def to_markdown(self) -> str:
        """Generate Markdown formatted summary."""
        lines = ["# Kernel Summary\n"]
        
        def add_component(comp: KernelComponent, level: int = 2):
            header = "#" * level
            lines.append(f"{header} {comp.kernel_type}\n")
            
            if comp.parameters:
                lines.append("| Parameter | Value |")
                lines.append("|-----------|-------|")
                for k, v in comp.parameters.items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")
            
            if comp.interpretations:
                lines.append("**Interpretations:**")
                for k, v in comp.interpretations.items():
                    lines.append(f"- **{k}**: {v}")
                lines.append("")
            
            for child in comp.children:
                add_component(child, level + 1)
        
        add_component(self.root)
        return "\n".join(lines)
    
    def to_json(self, indent: int = 2) -> str:
        """Generate JSON representation."""
        return json.dumps(self.root.to_dict(), indent=indent)


# Register specific kernel interpreters
@KernelInterpreter.register_kernel("RBF")
def _interpret_rbf(kernel: Any, config: InterpretationConfig) -> Dict[str, Any]:
    """Specialized RBF kernel interpretation."""
    result = {"parameters": {}, "interpretations": {}, "is_composite": False}
    
    # RBF-specific: lengthscale is crucial
    ls = float(kernel.lengthscale)
    result["parameters"]["lengthscale"] = ls
    
    category = config.lengthscale.categorize(ls)
    if category == SmoothnessCategory.RAPID_VARIATION:
        advice = "Model will fit noise - consider increasing"
    elif category == SmoothnessCategory.SMOOTH_TREND:
        advice = "Model may underfit - consider decreasing"
    else:
        advice = "Well-balanced flexibility"
    
    result["interpretations"]["smoothness"] = f"{category.describe()}. {advice}"
    
    # Variance
    var = float(kernel.variance)
    result["parameters"]["variance"] = var
    var_cat = config.variance.categorize(var)
    result["interpretations"]["signal_strength"] = var_cat.describe("Signal")
    
    return result


@KernelInterpreter.register_kernel("Linear")
def _interpret_linear(kernel: Any, config: InterpretationConfig) -> Dict[str, Any]:
    """Specialized Linear kernel interpretation."""
    result = {"parameters": {}, "interpretations": {}, "is_composite": False}
    
    if hasattr(kernel, "variances"):
        variances = KernelInterpreter._extract_values(kernel.variances)
        result["parameters"]["ARD_variances"] = variances
        active_dims = sum(1 for v in np.atleast_1d(variances) if v > 0.1)
        result["interpretations"]["relevance"] = (
            f"Linear trend in {active_dims}/{len(np.atleast_1d(variances))} dimensions"
        )
    
    return result


@KernelInterpreter.register_kernel("PeriodicExponential")
@KernelInterpreter.register_kernel("PeriodicMatern32")
@KernelInterpreter.register_kernel("PeriodicMatern52")
def _interpret_periodic(kernel: Any, config: InterpretationConfig) -> Dict[str, Any]:
    """Specialized periodic kernel interpretation."""
    result = {"parameters": {}, "interpretations": {}, "is_composite": False}
    
    period = float(kernel.periodicity)
    result["parameters"]["period"] = period
    result["interpretations"]["pattern"] = f"Repeating pattern every {period:.2f} units"
    
    ls = float(kernel.lengthscale)
    result["parameters"]["decay_lengthscale"] = ls
    if ls > period * 2:
        result["interpretations"]["stability"] = "Long-range periodic correlations"
    else:
        result["interpretations"]["stability"] = "Local periodic patterns only"
    
    return result


# High-level API functions
def summarize_kernel(
    model: Any,
    X: Optional[np.ndarray] = None,
    verbose: bool = True,
    config: Optional[InterpretationConfig] = None,
    format: str = "text",
) -> Union[str, Dict[str, Any]]:
    """
    Generate comprehensive kernel interpretation.
    
    Args:
        model: GPy model with 'kern' attribute
        X: Training data (optional, for context-aware scaling)
        verbose: Print summary if True
        config: Interpretation configuration
        format: Output format ('text', 'markdown', 'json', 'dict')
        
    Returns:
        Formatted string or dictionary depending on format
        
    Raises:
        KernelError: If model invalid or kernel uninterpretable
    """
    if not hasattr(model, "kern"):
        raise KernelError("Model must have 'kern' attribute")
    
    # Auto-adjust config based on data scale if provided
    cfg = config or InterpretationConfig()
    if X is not None:
        cfg = _adapt_config_to_data(cfg, X)
    
    # Build interpretation tree
    interpreter = KernelInterpreter(cfg)
    try:
        root = interpreter.interpret(model.kern)
    except Exception as e:
        raise KernelError(f"Failed to interpret kernel: {e}") from e
    
    # Format output
    formatter = KernelSummaryFormatter(root)
    
    if format == "dict":
        return root.to_dict()
    elif format == "json":
        result = formatter.to_json()
    elif format == "markdown":
        result = formatter.to_markdown()
    else:  # text
        result = formatter.to_text()
    
    if verbose and format in ("text", "markdown"):
        print(result)
    
    return result


def interpret_lengthscale(
    lengthscale: Union[float, np.ndarray, List[float]],
    config: Optional[LengthscaleThresholds] = None,
    return_category: bool = False,
) -> Union[str, Tuple[str, SmoothnessCategory]]:
    """
    Interpret lengthscale magnitude with data-aware thresholds.
    
    Args:
        lengthscale: Single value or array of lengthscales
        config: Threshold configuration
        return_category: If True, return (description, category) tuple
        
    Returns:
        Interpretation string, or tuple if return_category=True
    """
    cfg = config or LengthscaleThresholds()
    
    # Normalize input
    if isinstance(lengthscale, (list, np.ndarray)):
        arr = np.atleast_1d(lengthscale)
        mean_ls = float(np.mean(arr))
        range_str = f"[{np.min(arr):.2f}, {np.max(arr):.2f}]"
        is_ard = len(arr) > 1
    else:
        mean_ls = float(lengthscale)
        range_str = f"{mean_ls:.2f}"
        is_ard = False
    
    category = cfg.categorize(mean_ls)
    description = category.describe()
    
    if is_ard:
        description += f" (ARD, range: {range_str})"
    else:
        description += f" ({range_str})"
    
    if return_category:
        return description, category
    return description


def interpret_variance(
    variance: float,
    name: str = "Signal",
    config: Optional[VarianceThresholds] = None,
    return_category: bool = False,
) -> Union[str, Tuple[str, VarianceCategory]]:
    """
    Interpret variance magnitude with context-aware messaging.
    
    Args:
        variance: Variance value
        name: Type of variance ("Signal" or "Noise")
        config: Threshold configuration
        return_category: If True, return (description, category) tuple
        
    Returns:
        Interpretation string, or tuple if return_category=True
    """
    cfg = config or VarianceThresholds()
    category = cfg.categorize(variance)
    description = category.describe(name) + f" (≈{variance:.3f})"
    
    if return_category:
        return description, category
    return description


def format_kernel_tree(model: Any, style: str = "unicode") -> str:
    """
    Pretty-print kernel tree structure.
    
    Args:
        model: GPy model
        style: Output style ('unicode', 'ascii', 'minimal')
        
    Returns:
        Formatted tree string
    """
    if not hasattr(model, "kern"):
        raise KernelError("Model must have 'kern' attribute")
    
    interpreter = KernelInterpreter()
    root = interpreter.interpret(model.kern)
    
    if style == "minimal":
        return root.kernel_type
    
    # Use formatter's tree rendering
    formatter = KernelSummaryFormatter(root)
    # Extract just the tree portion
    full_text = formatter.to_text(verbose=False)
    # Find and return just the structure section
    lines = full_text.split("\n")
    start_idx = None
    for i, line in enumerate(lines):
        if "Structure:" in line:
            start_idx = i + 1
        elif start_idx and line.startswith("Components:"):
            return "\n".join(lines[start_idx:i]).strip()
    
    return root.kernel_type


def count_kernel_components(model: Any) -> int:
    """Count total number of kernel components (leaf nodes)."""
    if not hasattr(model, "kern"):
        return 0
    
    def count_leaves(kernel: Any) -> int:
        if hasattr(kernel, "parts") and kernel.parts:
            return sum(count_leaves(k) for k in kernel.parts)
        return 1
    
    return count_leaves(model.kern)


def extract_kernel_params_flat(model: Any) -> Dict[str, float]:
    """
    Extract all kernel parameters as flat dictionary with dotted paths.
    
    Args:
        model: GPy model
        
    Returns:
        Flat dictionary mapping "path.param" to value
    """
    if not hasattr(model, "kern"):
        raise KernelError("Model must have 'kern' attribute")
    
    params = {}
    
    def extract(kernel: Any, path: str = ""):
        current_path = f"{path}.{kernel.name}" if path else kernel.name
        
        if hasattr(kernel, "parameters"):
            for param in kernel.parameters:
                param_path = f"{current_path}.{param.name}"
                val = KernelInterpreter._extract_values(param)
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        params[f"{param_path}[{i}]"] = float(v)
                else:
                    params[param_path] = float(val)
        
        if hasattr(kernel, "parts") and kernel.parts:
            for i, part in enumerate(kernel.parts):
                extract(part, current_path)
    
    extract(model.kern)
    return params


def get_lengthscale(model: Any, as_dict: bool = False) -> Union[float, Dict[str, float]]:
    """
    Extract lengthscale(s) from model kernel.
    
    Args:
        model: GPy model
        as_dict: Return dictionary with component paths as keys
        
    Returns:
        Single float, array, or dictionary of lengthscales
    """
    if not hasattr(model, "kern"):
        raise KernelError("Model must have 'kern' attribute")
    
    if as_dict:
        result = {}
        def find_lengthscales(kernel: Any, path: str = ""):
            current = f"{path}.{kernel.name}" if path else kernel.name
            if hasattr(kernel, "lengthscale"):
                result[current] = KernelInterpreter._extract_values(kernel.lengthscale)
            if hasattr(kernel, "parts") and kernel.parts:
                for part in kernel.parts:
                    find_lengthscales(part, current)
        find_lengthscales(model.kern)
        return result
    else:
        # Return first found lengthscale
        if hasattr(model.kern, "lengthscale"):
            val = KernelInterpreter._extract_values(model.kern.lengthscale)
            return val[0] if isinstance(val, list) else val
        raise KernelError("Model kernel has no lengthscale attribute")


def get_noise_variance(model: Any) -> float:
    """Extract noise variance from GP model."""
    if not hasattr(model, "likelihood"):
        raise KernelError("Model must have 'likelihood' attribute")
    try:
        return float(model.likelihood.variance)
    except Exception as e:
        raise KernelError(f"Could not extract noise variance: {e}") from e


# Private utilities
def _adapt_config_to_data(
    config: InterpretationConfig,
    X: np.ndarray,
) -> InterpretationConfig:
    """
    Auto-scale thresholds based on data characteristics.
    
    Args:
        config: Base configuration
        X: Training data (n_samples, n_dims)
        
    Returns:
        Adapted configuration
    """
    if X.ndim != 2 or X.shape[0] < 2:
        return config
    
    # Compute data scale
    ranges = np.ptp(X, axis=0)  # Peak-to-peak (max - min)
    median_range = float(np.median(ranges[ranges > 0]))
    
    if median_range <= 0:
        return config
    
    # Scale thresholds proportionally to data range
    scale_factor = median_range / 2.0  # Assuming standardized data ~2 range
    
    new_ls = LengthscaleThresholds(
        rapid_variation=config.lengthscale.rapid_variation * scale_factor,
        smooth_trend=config.lengthscale.smooth_trend * scale_factor,
    )
    
    return InterpretationConfig(
        lengthscale=new_ls,
        variance=config.variance,
    )