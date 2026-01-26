"""
Data influence analysis: quantify how training points affect model uncertainty.
"""

import numpy as np
from typing import Dict, Any, Tuple
import GPy
from scipy.linalg import solve_triangular

class DataInfluenceMap:
    """
    Compute influence scores for training data points.
    
    Identifies which observations most reduce model uncertainty and
    which have high leverage on predictions.
    """
    
    def __init__(self, model: GPy.models.GPRegression):
        """
        Initialize with GP model.
        
        Args:
            model: Trained GPy model
        """
        self.model = model
    
    def compute_influence_scores(self, X_train: np.ndarray) -> np.ndarray:
        """
        Approximate influence scores using leverage scores (fast).
        
        Leverage score = diagonal of K(K + σ²I)⁻¹K, approximated via
        Cholesky decomposition for numerical stability.
        
        Args:
            X_train: Training input locations
            
        Returns:
            Influence scores array (higher = more influential)
        """
        # Compute covariance matrix
        K = self.model.kern.K(X_train, X_train)
        noise_var = np.exp(self.model.likelihood.log_variance)
        K_stable = K + np.eye(K.shape[0]) * noise_var
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K_stable)
        except np.linalg.LinAlgError:
            # Add jitter if needed
            K_stable += np.eye(K.shape[0]) * 1e-6
            L = np.linalg.cholesky(K_stable)
        
        # Solve K⁻¹ for each basis vector (leverage scores)
        n = X_train.shape[0]
        scores = np.zeros(n)
        
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            
            # Solve linear system efficiently
            v = solve_triangular(L, e_i, lower=True)
            inv_row = solve_triangular(L, v, lower=True, trans='T')
            
            # Leverage score is inverse of diagonal element
            scores[i] = 1.0 / (inv_row[i] + 1e-10)
        
        return scores
    
    def compute_loo_variance_increase(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact Leave-One-Out variance increase (slower but precise).
        
        Args:
            X_train: Training inputs
            y_train: Training outputs
            
        Returns:
            Tuple of (variance_increase, prediction_errors)
        """
        n = X_train.shape[0]
        variance_increase = np.zeros(n)
        prediction_errors = np.zeros(n)
        
        # Pre-compute full covariance for efficiency
        K_full = self.model.kern.K(X_train, X_train)
        noise_var = np.exp(self.model.likelihood.log_variance)
        np.fill_diagonal(K_full, np.diag(K_full) + noise_var)
        
        for i in range(n):
            # Leave-one-out indices
            idx = np.arange(n) != i
            
            try:
                # Sub-covariance matrix
                K_loo = K_full[np.ix_(idx, idx)]
                k_star = K_full[np.ix_([i], idx)][0]
                
                # Posterior variance at x_i without it in training
                L = np.linalg.cholesky(K_loo)
                v = solve_triangular(L, k_star, lower=True)
                var_without_i = K_full[i, i] - np.dot(v, v)
                
                # Variance with x_i in training (from model)
                _, var_with_i = self.model.predict(X_train[i:i+1])
                
                variance_increase[i] = max(0, var_without_i - var_with_i[0, 0])
                
                # LOO prediction error
                mean_loo, _ = self.model.predict(X_train[i:i+1])
                prediction_errors[i] = abs(mean_loo[0, 0] - y_train[i])
                
            except Exception:
                variance_increase[i] = np.nan
                prediction_errors[i] = np.nan
        
        return variance_increase, prediction_errors
    
    def plot_influence(
        self, 
        X_train: np.ndarray, 
        influence_scores: np.ndarray,
        ax=None,
        **scatter_kwargs
    ) -> plt.Axes:
        """
        Visualize data point influence.
        
        Args:
            X_train: Training input locations
            influence_scores: Computed influence scores
            ax: Matplotlib axes
            **scatter_kwargs: Additional scatter plot arguments
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Normalize scores for visual size
        sizes = 100 + (influence_scores / np.max(influence_scores)) * 400
        
        scatter = ax.scatter(
            X_train.flatten(), 
            np.zeros_like(X_train.flatten()),
            c=influence_scores,
            s=sizes,
            cmap=scatter_kwargs.get('cmap', 'viridis'),
            alpha=scatter_kwargs.get('alpha', 0.7),
            edgecolors='black',
            linewidth=1
        )
        
        ax.set_xlabel("Input Space", fontsize=12)
        ax.set_title("Data Point Influence Map\n(size ∝ influence)", 
                    fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Influence Score', fontsize=10)
        
        # Remove y-axis ticks (all zero)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        return ax
    
    def get_influence_report(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive influence analysis report.
        
        Args:
            X_train: Training inputs
            y_train: Training outputs
            
        Returns:
            Dictionary with influence statistics and diagnostics
        """
        scores = self.compute_influence_scores(X_train)
        loo_var, loo_err = self.compute_loo_variance_increase(X_train, y_train)
        
        # Find most/least influential
        most_inf_idx = np.argmax(scores)
        least_inf_idx = np.argmin(scores)
        
        return {
            "influence_scores": scores,
            "loo_variance_increase": loo_var,
            "loo_prediction_errors": loo_err,
            "statistics": {
                "mean_influence": float(np.mean(scores)),
                "std_influence": float(np.std(scores)),
                "max_influence": float(np.max(scores)),
                "min_influence": float(np.min(scores)),
            },
            "most_influential_point": {
                "index": int(most_inf_idx),
                "location": X_train[most_inf_idx].flatten(),
                "score": float(scores[most_inf_idx]),
                "loo_error": float(loo_err[most_inf_idx]) if not np.isnan(loo_err[most_inf_idx]) else None
            },
            "least_influential_point": {
                "index": int(least_inf_idx),
                "location": X_train[least_inf_idx].flatten(),
                "score": float(scores[least_inf_idx]),
                "loo_error": float(loo_err[least_inf_idx]) if not np.isnan(loo_err[least_inf_idx]) else None
            },
            "diagnostics": {
                "high_leverage_points": int(np.sum(scores > np.percentile(scores, 95))),
                "outliers": int(np.sum(loo_err > np.percentile(loo_err, 95))),
            }
        }
