"""
Basic Usage
===========

Introduction to GPClarity core features.
"""

import numpy as np
import GPy
import gpclarity

# Generate toy data
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * np.random.randn(50)

# Train GP
kernel = GPy.kern.RBF(1) + GPy.kern.White(1)
model = GPy.models.GPRegression(X, y[:, None], kernel)
model.optimize(messages=False)

# Kernel interpretation
print("=== Kernel Summary ===")
summary = gpclarity.summarize_kernel(model, verbose=True)

# Complexity analysis
print("\n=== Complexity Analysis ===")
report = gpclarity.compute_complexity_score(model, X)
print(f"Category: {report['category']}")
print(f"Risk level: {report['risk_level']}")

# Uncertainty profiling
print("\n=== Uncertainty Analysis ===")
profiler = gpclarity.UncertaintyProfiler(model, X_train=X)
X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
diag = profiler.compute_diagnostics(X_test)
print(f"Extrapolation points: {diag.n_extrapolation_points}")