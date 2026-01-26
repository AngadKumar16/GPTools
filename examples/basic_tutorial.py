"""
GPDiagnostics Basic Tutorial
============================

This tutorial demonstrates core functionality of gpdiagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
import gpdiagnostics
import GPy

# Generate synthetic data
np.random.seed(42)
X_train = np.linspace(0, 10, 40).reshape(-1, 1)
y_train = np.sin(X_train).flatten() + 0.1 * np.random.randn(40)

# Create and train GP model
kernel = GPy.kern.RBF(1, lengthscale=2.0) + GPy.kern.White(1, variance=0.01)
model = GPy.models.GPRegression(X_train, y_train[:, None], kernel)
model.optimize()

print("✅ Model trained successfully!")

# 1. Kernel Interpretation
print("\n" + "="*50)
print("KERNEL INTERPRETATION")
print("="*50)
gpdiagnostics.summarize_kernel(model)

# 2. Uncertainty Profiling
print("\n" + "="*50)
print("UNCERTAINTY ANALYSIS")
print("="*50)

profiler = gpdiagnostics.UncertaintyProfiler(model)
X_test = np.linspace(-2, 12, 200).reshape(-1, 1)

# Plot uncertainty profile
fig, ax = plt.subplots(figsize=(12, 6))
profiler.plot(X_test, X_train=X_train, y_train=y_train, ax=ax)
plt.savefig("uncertainty_profile.png", dpi=150, bbox_inches='tight')
plt.show()

# Compute diagnostics
diagnostics = profiler.compute_diagnostics(X_test)
print(f"Mean uncertainty: {diagnostics['mean_uncertainty']:.4f}")
print(f"Max uncertainty: {diagnostics['max_uncertainty']:.4f}")

# 3. Hyperparameter Tracking
print("\n" + "="*50)
print("HYPERPARAMETER TRACKING")
print("="*50)

tracker = gpdiagnostics.HyperparameterTracker(model)
print("Re-running optimization with tracking...")

# Re-initialize for demo
model.kern.lengthscale = 1.0
model.kern.white.variance = 0.1

history = tracker.wrapped_optimize(max_iters=20)
print(f"Tracked {len(history)} parameters over 20 iterations")

# Plot evolution
fig = tracker.plot_evolution(figsize=(12, 8))
plt.savefig("hyperparam_evolution.png", dpi=150, bbox_inches='tight')
plt.show()

# 4. Model Complexity
print("\n" + "="*50)
print("MODEL COMPLEXITY ANALYSIS")
print("="*50)

complexity = gpdiagnostics.compute_complexity_score(model, X_train)
print(f"Complexity Score: {complexity['score']:.3f}")
print(f"Interpretation: {complexity['interpretation']}")
print("Components:")
for key, val in complexity['components'].items():
    print(f"  └─ {key}: {val:.3f}")

# 5. Data Influence
print("\n" + "="*50)
print("DATA INFLUENCE ANALYSIS")
print("="*50)

influence = gpdiagnostics.DataInfluenceMap(model)
scores = influence.compute_influence_scores(X_train)

fig, ax = plt.subplots(figsize=(12, 6))
influence.plot_influence(X_train, scores, ax=ax)
plt.savefig("data_influence.png", dpi=150, bbox_inches='tight')
plt.show()

# Get detailed report
report = influence.get_influence_report(X_train, y_train)
most_inf = report["most_influential_point"]
print(f"Most influential point: index {most_inf['index']} (score: {most_inf['score']:.3f})")

print("\n🎉 Tutorial complete! Check the generated PNG files for visualizations.")
