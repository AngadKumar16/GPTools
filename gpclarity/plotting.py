"""
Visualization utilities for GP interpretability.
"""

from typing import Optional

import numpy as np

def plot_influence_map(
    X_train: np.ndarray,
    influence_scores: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    title: str = "Data Point Influence Map",
    **scatter_kwargs,
):
    """
    Visualize data point influence in input space.
    
    Args:
        X_train: Training inputs (shape: [n, d] where d <= 2)
        influence_scores: Influence scores per point
        ax: Matplotlib axes
        title: Plot title
        **scatter_kwargs: Passed to ax.scatter
        
    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as e:
        raise ImportError(
            "plotting requires matplotlib. Install: pip install matplotlib"
        ) from e

    # Dimensionality check
    if X_train.shape[1] > 2:
        raise ValueError(
            f"Cannot plot {X_train.shape[1]}D data directly. "
            "Use PCA reduction or select 2 dimensions."
        )
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Handle non-finite scores
    safe_scores = np.where(np.isfinite(influence_scores), influence_scores, 0.0)
    
    # Normalize sizes
    max_score = np.max(safe_scores) if np.max(safe_scores) > 0 else 1.0
    sizes = 50 + (safe_scores / (max_score + 1e-10)) * 500
    
    # Color normalization
    norm = Normalize(vmin=np.min(safe_scores), vmax=max_score)
    
    scatter = ax.scatter(
        X_train[:, 0],
        X_train[:, 1] if X_train.shape[1] > 1 else np.zeros(X_train.shape[0]),
        c=safe_scores,
        s=sizes,
        cmap=scatter_kwargs.get("cmap", "viridis"),
        alpha=scatter_kwargs.get("alpha", 0.7),
        norm=norm,
        edgecolors="black",
        linewidth=0.5,
    )
    
    ax.set_xlabel("Dimension 1", fontsize=11)
    if X_train.shape[1] > 1:
        ax.set_ylabel("Dimension 2", fontsize=11)
    else:
        ax.set_ylabel("Zero baseline (1D projection)", fontsize=11)
        
    ax.set_title(title + "\n(size ∝ influence)", fontsize=12, fontweight="bold")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Influence Score", fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    return ax



def plot_optimization_trajectory(
    tracker: "HyperparameterTracker",
    params: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_convergence: bool = True,
    show_ll: bool = True,
    n_cols: int = 2,
) -> "plt.Figure":
    """
    Plot parameter trajectories from optimization history.
    
    Args:
        tracker: HyperparameterTracker with recorded history
        params: Specific parameters to plot (all if None)
        figsize: Figure dimensions
        show_convergence: Show final value and convergence bands
        show_ll: Include log-likelihood subplot
        n_cols: Number of columns in subplot grid
        
    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "plotting requires matplotlib. Install: pip install matplotlib"
        ) from e

    history = tracker.history
    
    # Determine parameters to plot
    if params is None:
        params = list(history[0].parameters.keys())
    
    # Add log-likelihood to params if requested
    plot_items = list(params)
    if show_ll:
        has_ll = any(s.log_likelihood is not None for s in history)
        if has_ll:
            plot_items.append("__log_likelihood__")
    
    # Calculate grid layout
    n_plots = len(plot_items)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, param_name in enumerate(plot_items):
        ax = axes_flat[idx]
        
        if param_name == "__log_likelihood__":
            iterations = [s.iteration for s in history]
            values = [s.log_likelihood for s in history]
            ax.plot(iterations, values, linewidth=2, color="green")
            ax.set_title("Log Likelihood", fontweight="bold")
            ax.set_ylabel("LL")
        else:
            iterations, values = tracker.get_parameter_trajectory(param_name)
            
            # Handle multi-dimensional
            if values.ndim > 1:
                for dim in range(values.shape[1]):
                    ax.plot(iterations, values[:, dim], 
                           label=f"Dim {dim}", alpha=0.8)
                ax.legend(fontsize=8)
            else:
                ax.plot(iterations, values, linewidth=2, color="#2E86AB")
                
                if show_convergence and len(values) > 10:
                    final_val = values[-1]
                    ax.axhline(y=final_val, color="green", linestyle="--", 
                              alpha=0.5, label=f"Final: {final_val:.3f}")
                    # Convergence band (±1 std of last 10%)
                    window = max(5, len(values) // 10)
                    recent_std = np.std(values[-window:])
                    ax.fill_between(iterations, 
                                   final_val - recent_std, 
                                   final_val + recent_std,
                                   alpha=0.1, color="green")
                    ax.legend(fontsize=8)
            
            ax.set_title(f"{param_name.replace('_', ' ').title()}", 
                        fontweight="bold")
            ax.set_ylabel("Value")
        
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(plot_items), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle("Optimization Trajectories", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    return fig

def plot_uncertainty_profile(
    profiler: "UncertaintyProfiler",
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
    color_test: str = "green",
    show_regions: bool = False,
    **kwargs,
) -> "plt.Axes":
    """
    Comprehensive uncertainty profile visualization.
    
    Args:
        profiler: UncertaintyProfiler instance
        X_test: Test locations
        X_train: Training inputs
        y_train: Training outputs
        y_test: Test ground truth (optional)
        ax: Matplotlib axes
        confidence_levels: Sigma levels for confidence bands
        plot_std: Overlay standard deviation as line
        fill_alpha: Opacity of confidence bands
        color_mean: Color for mean line
        color_fill: Color for uncertainty bands
        color_train: Color for training points
        color_test: Color for test ground truth
        show_regions: Highlight extrapolation regions
        
    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as e:
        raise ImportError(
            "plotting requires matplotlib. Install: pip install matplotlib"
        ) from e

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    # Get predictions
    pred = profiler.predict(X_test)
    mean, std = pred.mean, pred.std
    
    X_flat = X_test.flatten()
    mean_flat = mean.flatten()
    std_flat = std.flatten()

    # Sort for clean plotting (if 1D)
    if X_test.shape[1] == 1:
        sort_idx = np.argsort(X_flat)
        X_flat = X_flat[sort_idx]
        mean_flat = mean_flat[sort_idx]
        std_flat = std_flat[sort_idx]

    # Plot confidence bands (outer first)
    for level in sorted(confidence_levels, reverse=True):
        alpha = fill_alpha * (0.5 ** (list(confidence_levels).index(level)))
        ax.fill_between(
            X_flat,
            mean_flat - level * std_flat,
            mean_flat + level * std_flat,
            alpha=alpha,
            color=color_fill,
            label=f"±{level}σ" if level == max(confidence_levels) else None,
            zorder=1,
        )

    # Mean prediction
    ax.plot(
        X_flat,
        mean_flat,
        color=color_mean,
        linewidth=2.5,
        label="GP Mean",
        zorder=3,
    )

    # Standard deviation line (optional)
    if plot_std:
        ax.plot(
            X_flat,
            std_flat,
            color=color_fill,
            linestyle="--",
            alpha=0.7,
            label="Std Dev",
            zorder=2,
        )

    # Training data
    if X_train is not None and y_train is not None:
        ax.scatter(
            X_train.flatten(),
            y_train.flatten(),
            color=color_train,
            s=60,
            zorder=5,
            label="Training Data",
            edgecolors="white",
            linewidth=1,
            alpha=0.9,
        )

    # Test ground truth
    if y_test is not None:
        ax.scatter(
            X_test.flatten(),
            y_test.flatten(),
            color=color_test,
            s=40,
            marker="x",
            zorder=4,
            label="Ground Truth",
            alpha=0.7,
        )

    # Region highlighting
    if show_regions and X_train is not None:
        regions = profiler.classify_regions(X_test)
        
        # Highlight extrapolation regions
        ext_mask = regions == UncertaintyRegion.EXTRAPOLATION
        if np.any(ext_mask):
            ax.axvspan(
                np.min(X_test[ext_mask].flatten()),
                np.max(X_test[ext_mask].flatten()),
                alpha=0.1,
                color="red",
                label="Extrapolation",
            )

    ax.set_xlabel("Input", fontsize=12)
    ax.set_ylabel("Output", fontsize=12)
    ax.set_title("Uncertainty Profile", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    return ax