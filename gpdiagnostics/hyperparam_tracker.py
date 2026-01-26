"""
Hyperparameter evolution tracking during GP model optimization.
"""

import numpy as np
from typing import Dict, List, Callable, Optional
import GPy
import matplotlib.pyplot as plt

class HyperparameterTracker:
    """
    Track hyperparameter trajectories during model optimization.
    
    Provides tools for monitoring convergence, visualizing parameter
    evolution, and detecting optimization issues.
    """
    
    def __init__(self, model: GPy.models.GPRegression):
        """
        Initialize tracker with GP model.
        
        Args:
            model: GPy model to track
        """
        self.model = model
        self.history: Dict[str, List[float]] = {}
        self.iteration_count = 0
        
    def record_state(self):
        """Snapshot current hyperparameter values."""
        for param in self.model.parameters:
            name = param.name
            if name not in self.history:
                self.history[name] = []
            
            val = param.values
            if hasattr(val, '__iter__'):
                # Handle vector parameters (e.g., ARD lengthscales)
                self.history[name].append(val.copy())
            else:
                self.history[name].append(float(val))
    
    def wrapped_optimize(
        self,
        max_iters: int = 100,
        callback: Optional[Callable] = None,
        **optimize_kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform optimization while tracking parameters at each iteration.
        
        Args:
            max_iters: Maximum optimization iterations
            callback: Optional function(model, iteration, history) called each step
            **optimize_kwargs: Arguments passed to model.optimize()
            
        Returns:
            Complete parameter history dictionary
        """
        self.history = {p.name: [] for p in self.model.parameters}
        
        for i in range(max_iters):
            try:
                # Single iteration optimization
                self.model.optimize(max_iters=1, **optimize_kwargs)
                self.record_state()
                self.iteration_count += 1
                
                if callback:
                    callback(self.model, i, self.history)
                    
            except Exception as e:
                print(f"⚠️ Optimization failed at iteration {i}: {e}")
                break
        
        return self.history
    
    def plot_evolution(
        self, 
        params: Optional[List[str]] = None,
        figsize: tuple = (12, 8),
        show_convergence: bool = True
    ) -> plt.Figure:
        """
        Plot parameter trajectories with optional convergence indicators.
        
        Args:
            params: List of parameter names to plot (plots all if None)
            figsize: Figure size tuple
            show_convergence: Whether to show convergence markers
            
        Returns:
            Matplotlib figure object
        """
        if not self.history:
            raise ValueError("No optimization history recorded")
        
        plot_params = params if params else list(self.history.keys())
        
        fig, axes = plt.subplots(
            len(plot_params), 
            1, 
            figsize=figsize,
            squeeze=False
        )
        axes = axes.flatten()
        
        for idx, name in enumerate(plot_params):
            if name not in self.history:
                continue
            
            values = self.history[name]
            ax = axes[idx]
            
            # Handle multi-dimensional parameters
            if len(values) > 0 and hasattr(values[0], '__iter__'):
                values_arr = np.array(values)
                for dim in range(values_arr.shape[1]):
                    ax.plot(values_arr[:, dim], label=f"Dim {dim}", alpha=0.8)
                ax.legend()
            else:
                ax.plot(values, linewidth=2.5, color='#2E86AB')
            
            # Convergence markers
            if show_convergence and len(values) > 10:
                final_val = values[-1]
                ax.axhline(y=final_val, color='green', linestyle='--', 
                          alpha=0.5, label=f'Final: {final_val:.3f}')
            
            ax.set_title(f"{name.replace('_', ' ').title()} Evolution", 
                        fontweight='bold')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()
        
        plt.suptitle("Hyperparameter Optimization Trajectories", 
                    fontsize=16, y=0.995)
        plt.tight_layout()
        return fig
    
    def get_convergence_report(self, window: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Analyze parameter convergence with statistical metrics.
        
        Args:
            window: Number of iterations for convergence analysis
            
        Returns:
            Dictionary with convergence statistics per parameter
        """
        report = {}
        for name, values in self.history.items():
            if len(values) < window * 2:
                continue
            
            # Recent vs earlier window
            recent = np.array(values[-window:])
            earlier = np.array(values[:window])
            
            report[name] = {
                "initial_mean": float(np.mean(earlier)),
                "final_mean": float(np.mean(recent)),
                "relative_change": float(abs(np.mean(recent) - np.mean(earlier)) / 
                                       (abs(np.mean(earlier)) + 1e-10)),
                "final_std": float(np.std(recent)),
                "converged": float(np.std(recent)) < 0.01 * float(np.mean(np.abs(recent)))
            }
        
        return report
