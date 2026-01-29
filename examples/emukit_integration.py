"""
Advanced Integration with Emukit for Bayesian Optimization
===========================================================

This example shows how to use gpclarity with emukit's 
experimental design loops.
"""

import gpclarity
import numpy as np
import matplotlib.pyplot as plt
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop import UserFunctionWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from gpclarity import UncertaintyProfiler, HyperparameterTracker
from gpclarity import ClarityBayesianOptimizationLoop

# Define objective function
def objective_function(x):
    """Forrester function (1D test function)."""
    return (6*x - 2)**2 * np.sin(12*x - 4)

# Parameter space
space = ParameterSpace([ContinuousParameter('x', 0, 1)])

# Initial data
X_init = np.array([[0.1], [0.5], [0.9]])
Y_init = objective_function(X_init)

# Create GPy model
kernel = GPy.kern.RBF(1, lengthscale=0.1, variance=1.0) + GPy.kern.White(1, variance=1e-6)
gpy_model = GPy.models.GPRegression(X_init, Y_init, kernel)
gpy_model.Gaussian_noise.variance.fix(1e-6)

# Create emukit model
from emukit.model_wrappers import GPyModelWrapper
emukit_model = GPyModelWrapper(gpy_model)

# Create acquisition
acquisition = ExpectedImprovement(emukit_model)

# Create enhanced loop with diagnostics
loop = ClarityBayesianOptimizationLoop(
    model=emukit_model,
    space=space,
    acquisition=acquisition
)

# Run optimization
loop.run_loop(
    user_function=UserFunctionWrapper(objective_function),
    stopping_condition=lambda loop: loop.loop_state.iteration >= 15
)

print(f"Optimization completed with {loop.loop_state.iteration} iterations")

# Access accumulated diagnostics
print("\n" + "="*60)
print("OPTIMIZATION DIAGNOSTICS SUMMARY")
print("="*60)

# Plot diagnostics evolution
fig = loop.plot_diagnostics()
plt.suptitle("Bayesian Optimization Diagnostics Evolution", 
             fontsize=16, y=0.995)
plt.savefig("bo_diagnostics.png", dpi=150, bbox_inches='tight')
plt.show()

# Analyze final model
print("\nFinal Model Analysis:")
print("-" * 30)

final_model = loop.model.model
gpclarity.summarize_kernel(final_model)

# Uncertainty at final acquisition points
X_all = loop.loop_state.X
X_candidate = np.linspace(0, 1, 200).reshape(-1, 1)

profiler = UncertaintyProfiler(final_model)
regions = profiler.identify_uncertainty_regions(X_candidate, threshold_percentile=90)

print(f"\nHigh uncertainty points identified: {len(regions['high_uncertainty_points'])}")
print(f"Threshold value: {regions['threshold']:.4f}")

# Compare influence of initial vs acquired points
influence = gpclarity.DataInfluenceMap(final_model)
scores = influence.compute_influence_scores(X_all)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(
    X_all.flatten(), 
    np.zeros_like(X_all.flatten()),
    c=scores,
    s=100 + scores / np.max(scores) * 400,
    cmap='viridis',
    alpha=0.7,
    edgecolors='white',
    linewidth=2
)

# Mark initial points
ax.scatter(
    X_init.flatten(),
    np.zeros_like(X_init.flatten()),
    marker='s',
    s=200,
    facecolors='none',
    edgecolors='red',
    linewidth=3,
    label='Initial Design'
)

ax.set_xlabel("Input Space", fontsize=12)
ax.set_title("Data Point Influence\n(red squares = initial design)", 
            fontsize=14, fontweight='bold')
plt.colorbar(ax.collections[0], label='Influence Score')
plt.savefig("bo_data_influence.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n🎯 Emukit integration demo complete!")
