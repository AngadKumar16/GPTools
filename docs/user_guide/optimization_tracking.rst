Optimization Tracking
=====================

Monitor Hyperparameter Convergence
----------------------------------

GP optimization can fail silently. GPClarity tracks hyperparameter trajectories to ensure reliable convergence.

Basic Usage
-----------

.. code-block:: python

   from gpclarity import HyperparameterTracker
   
   tracker = HyperparameterTracker(model)
   
   # Optimize with tracking
   history = tracker.wrapped_optimize(max_iters=100)
   
   # Check convergence
   report = tracker.get_convergence_report(window=10)
   
   for param, metrics in report.items():
       status = "✓" if metrics.is_converged else "✗"
       print(f"{status} {param}: {metrics.trend_direction}")

Understanding Convergence Metrics
---------------------------------

For each parameter, GPClarity reports:

* **Initial vs final mean**: Average value in first/last window
* **Relative change**: Normalized difference
* **Final standard deviation**: Variability in final iterations
* **Coefficient of variation**: CV < 1% suggests convergence
* **Trend direction**: Increasing, decreasing, or stable

Detecting Problems
------------------

Oscillation Detection
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   issues = tracker.detect_optimization_issues()
   
   for warning in issues['warnings']:
       print(f"⚠️ {warning}")
   
   for rec in issues['recommendations']:
       print(f"💡 {rec}")

Common issues detected:
- Non-finite parameter values
- High oscillation (no convergence)
- Decreasing log-likelihood
- Gradient explosions

Visualization
-------------

.. code-block:: python

   fig = tracker.plot_evolution(
       params=['rbf.variance', 'rbf.lengthscale'],
       show_convergence=True,
       n_cols=2,
   )

Shows:
- Parameter trajectories over iterations
- Convergence bands (±1 std of final window)
- Final values as reference lines

Advanced Tracking
-----------------

Custom Callbacks
^^^^^^^^^^^^^^^^

.. code-block:: python

   def my_callback(model, iteration, history):
       if iteration % 10 == 0:
           print(f"Iter {iteration}: LL={model.log_likelihood():.2f}")
   
   tracker.wrapped_optimize(
       max_iters=200,
       callback=my_callback,
       capture_every=5,  # Record every 5th iteration
   )

Early Stopping
^^^^^^^^^^^^^^

.. code-block:: python

   history = tracker.wrapped_optimize(
       max_iters=500,
       patience=20,  # Stop if no improvement for 20 iterations
       convergence_tolerance=1e-4,
   )

Export to DataFrame
-------------------

.. code-block:: python

   df = tracker.to_dataframe()
   
   # Save for external analysis
   df.to_csv('optimization_history.csv')
   
   # Analyze specific parameters
   print(df[['iteration', 'rbf_lengthscale', 'log_likelihood']])

Best Practices
--------------

1. **Always check convergence** before using optimized model
2. **Use patience parameter** to avoid wasted iterations
3. **Capture every N iterations** for long optimizations (memory efficiency)
4. **Inspect log-likelihood trend** to detect numerical issues

See Also
--------

- :class:`gpclarity.HyperparameterTracker` for API details
- :doc:`complexity_analysis` for post-optimization model assessment