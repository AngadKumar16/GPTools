Quick Start Guide
=================

5-Minute Tour
-------------

Train a GP and get instant insights:

.. code-block:: python

   import numpy as np
   import GPy
   import gpclarity

   # 1. Create toy data
   np.random.seed(42)
   X = np.linspace(0, 10, 50).reshape(-1, 1)
   y = np.sin(X).flatten() + 0.1 * np.random.randn(50)

   # 2. Build and train GP
   kernel = GPy.kern.RBF(1) + GPy.kern.White(1)
   model = GPy.models.GPRegression(X, y[:, None], kernel)
   model.optimize(messages=True)

   # 3. Get clarity
   print(gpclarity.summarize_kernel(model, verbose=True))
   
Output::
   
   ╔══════════════════════════════════════════════════════════╗
   ║                    KERNEL SUMMARY                        ║
   ╚══════════════════════════════════════════════════════════╝
   
   Configuration:
     Lengthscale: rapid<0.5, smooth>2.0
     Variance: very_low<0.01, high>10.0
   
   Structure:
   └── add
       ├── rbf (RBF)
       └── white (White)
   
   Components:
   【RBF】 parts[0]
     ├─ lengthscale: 1.2345
     ├─ variance: 0.9876
     └─ Moderate flexibility (1.23). Well-balanced flexibility.
   
   【White】 parts[1]
     ├─ variance: 0.0101
     └─ Very low noise (≈0.010).

Common Workflows
----------------

Assess Model Complexity
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from gpclarity import compute_complexity_score
   
   report = compute_complexity_score(model, X)
   print(f"Complexity: {report['category']}")
   # > Complexity: MODERATE
   # > Well-balanced complexity
   # > Risk level: LOW

Analyze Uncertainty
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from gpclarity import UncertaintyProfiler
   
   profiler = UncertaintyProfiler(model, X_train=X)
   
   # Test on new grid
   X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
   
   # Get diagnostics
   diag = profiler.compute_diagnostics(X_test)
   print(f"Extrapolation points: {diag.n_extrapolation_points}")
   # > Extrapolation points: 40
   
   # Visualize
   profiler.plot(X_test, X_train=X, y_train=y)
   
   # Identify concerning regions
   regions = profiler.identify_uncertainty_regions(X_test)
   print(f"High uncertainty ratio: {regions['high_uncertainty_ratio']:.2%}")

Track Optimization
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from gpclarity import HyperparameterTracker
   
   tracker = HyperparameterTracker(model)
   
   # Optimize with tracking
   history = tracker.wrapped_optimize(max_iters=100)
   
   # Check convergence
   report = tracker.get_convergence_report()
   for param, metrics in report.items():
       print(f"{param}: {'✓ converged' if metrics.is_converged else '✗ not converged'}")
   
   # Plot trajectories
   fig = tracker.plot_evolution()

Analyze Data Influence
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from gpclarity import DataInfluenceMap
   
   influence = DataInfluenceMap(model)
   
   # Fast leverage scores
   result = influence.compute_influence_scores(X)
   
   # Detailed leave-one-out analysis (slower)
   loo_var, loo_err = influence.compute_loo_variance_increase(X, y)
   
   # Get comprehensive report
   report = influence.get_influence_report(X, y)
   print(f"Most influential point: index {report['most_influential']['index']}")
   
   # Visualize
   influence.plot_influence(X, result)

Next Steps
----------

See :doc:`user_guide/index` for detailed tutorials on each module.