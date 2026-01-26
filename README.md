# GPDiagnostics: Gaussian Process Interpretability Toolkit

![Python Version](https://img.shields.io/python/v/gpdiagnostics)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build Status](https://github.com/yourusername/gpdiagnostics/workflows/CI/badge.svg)

**GPDiagnostics** is a production-ready library that transforms black-box Gaussian Process models into interpretable, debuggable, and trustworthy tools. Built on GPy and emukit, it provides human-readable insights into kernel behavior, uncertainty patterns, and model complexity.

---

## 🎯 Features

- 🔍 **Kernel Interpretation**: Translate raw kernel math into human meaning
- 📊 **Uncertainty Profiling**: Visualize and diagnose uncertainty behavior
- 📈 **Hyperparameter Tracking**: Monitor optimization dynamics in real-time
- 🧮 **Complexity Quantification**: Measure and prevent overfitting
- 🎯 **Data Influence Analysis**: Identify impactful training points
- 🔗 **Emukit Integration**: Seamless Bayesian optimization support

---

## 🚀 Quick Start

```python
import gpdiagnostics
import GPy
import numpy as np

# Train a Gaussian Process
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * np.random.randn(50)

kernel = GPy.kern.RBF(1) + GPy.kern.White(1)
model = GPy.models.GPRegression(X, y[:, None], kernel)
model.optimize()

# 1. Get kernel interpretation
summary = gpdiagnostics.summarize_kernel(model)
# 🔍 Kernel: Composite with 2 components
# 📦 RBF (lengthscale: 1.23) → Moderate flexibility

# 2. Analyze uncertainty
profiler = gpdiagnostics.UncertaintyProfiler(model)
X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
profiler.plot(X_test, X_train=X, y_train=y)

# 3. Track hyperparameters during training
tracker = gpdiagnostics.HyperparameterTracker(model)
history = tracker.wrapped_optimize(max_iters=50)
tracker.plot_evolution()

# 4. Compute complexity score
complexity = gpdiagnostics.compute_complexity_score(model, X)
print(f"Complexity: {complexity['score']:.2f} - {complexity['interpretation']}")
