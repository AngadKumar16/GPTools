# GPClarity: Gaussian Process Interpretability Toolkit

![Python Version](https://img.shields.io/python/v/gpclarity)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build Status](https://github.com/yourusername/gpclarity/workflows/CI/badge.svg)

**GPClarity** is a production-ready library that transforms black-box Gaussian Process models into interpretable, debuggable, and trustworthy tools. Built on GPy and emukit, it provides human-readable insights into kernel behavior, uncertainty patterns, and model complexity.

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
import gpclarity
import GPy
import numpy as np

# Train a Gaussian Process
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).flatten() + 0.1 * np.random.randn(50)

kernel = GPy.kern.RBF(1) + GPy.kern.White(1)
model = GPy.models.GPRegression(X, y[:, None], kernel)
model.optimize()

summary = gpclarity.summarize_kernel(model)

profiler = gpclarity.UncertaintyProfiler(model)
X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
profiler.plot(X_test, X_train=X, y_train=y)

tracker = gpclarity.HyperparameterTracker(model)
history = tracker.wrapped_optimize(max_iters=50)
tracker.plot_evolution()

complexity = gpclarity.compute_complexity_score(model, X)
print(f"Complexity: {complexity['score']:.2f} - {complexity['interpretation']}")
```

---

## 📦 Installation

### Stable Release
```bash
pip install gpclarity
```

### Development Version
```bash
git clone https://github.com/yourusername/gpclarity.git
cd gpclarity
pip install -e ".[dev]"
```

### Conda (coming soon)
```bash
conda install -c conda-forge gpclarity
```

---

## 🏗️ Architecture

```
gpclarity/
├── kernel_summary
├── uncertainty_analysis
├── hyperparam_tracker
├── model_complexity
├── data_influence
└── utils
```

---

## 🔬 Advanced Usage

### Emukit Integration

```python
from gpclarity import ClarityBayesianOptimizationLoop

loop = ClarityBayesianOptimizationLoop(model, space)
loop.run_loop(user_function, stopping_condition)
loop.plot_diagnostics()
```

### Batch Processing

```python
models = [model1, model2, model3]
reports = [gpclarity.summarize_kernel(m, verbose=False) for m in models]
```

---

## 📊 Example Outputs

### Kernel Summary

```
🔍 KERNEL SUMMARY
Structure: ['RBF', 'White']
Components: 2

📦 RBF (lengthscale)
  └─ lengthscale: 1.23
  💡 Moderate flexibility

📦 White (variance)
  └─ variance: 0.01
  💡 Low observation noise
```

### Complexity Report

```json
{
  "score": 2.34,
  "interpretation": "Moderate complexity (well-balanced)",
  "components": {
    "n_kernel_parts": 2,
    "roughness_score": 0.81,
    "noise_ratio": 4.5
  }
}
```

---

## 🎓 Citation

```bibtex
@software{gpclarity2024,
  title={gpclarity: Gaussian Process Interpretability Toolkit},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/gpclarity},
  version={0.1.0}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md.

---

## 📄 License

MIT License - see LICENSE file for details.
