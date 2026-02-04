Uncertainty Analysis
====================

Understanding Model Confidence
------------------------------

GPs provide predictive uncertainty, but raw variance values can be misleading. GPClarity helps you understand *where* and *why* your model is uncertain.

Basic Usage
-----------

.. code-block:: python

   from gpclarity import UncertaintyProfiler
   
   profiler = UncertaintyProfiler(model, X_train=X_train)
   
   # Analyze test points
   X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
   diagnostics = profiler.compute_diagnostics(X_test)
   
   print(f"Mean uncertainty: {diagnostics.mean_uncertainty:.3f}")
   print(f"Extrapolation points: {diagnostics.n_extrapolation_points}")

Types of Uncertainty
--------------------

GPClarity distinguishes between:

**Epistemic Uncertainty** (model doesn't know)
   - High in extrapolation regions
   - Can be reduced with more data
   - Identified by :meth:`classify_regions`

**Aleatoric Uncertainty** (inherent noise)
   - High near noisy observations
   - Cannot be reduced with more data
   - Estimated from likelihood variance

Detecting Extrapolation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   regions = profiler.classify_regions(X_test)
   
   for region in UncertaintyRegion:
       mask = regions == region
       print(f"{region.name}: {np.sum(mask)} points")
   
   # Output:
   # EXTRAPOLATION: 40 points
   # INTERPOLATION: 120 points
   # BOUNDARY: 40 points

Calibration Checking
^^^^^^^^^^^^^^^^^^^^

Is your uncertainty well-calibrated? A 95% confidence interval should contain 95% of observations.

.. code-block:: python

   # Use validation data
   calibration = profiler.calibrate_uncertainty(X_val, y_val)
   
   print(f"Empirical coverage: {calibration['empirical_coverage']:.2%}")
   print(f"Optimal scale: {calibration['optimal_scale']:.2f}")
   
   # If scale > 1, your uncertainty is too small (overconfident)
   # If scale < 1, your uncertainty is too large (underconfident)

Visualizing Uncertainty
-----------------------

.. code-block:: python

   ax = profiler.plot(
       X_test,
       X_train=X_train,
       y_train=y_train,
       confidence_levels=(1.0, 2.0),
       show_regions=True,
   )

This shows:
- Mean prediction (blue line)
- 1-sigma and 2-sigma confidence bands (shaded)
- Training data points (red)
- Extrapolation regions (highlighted)

Identifying Problematic Regions
-------------------------------

Find where predictions are unreliable:

.. code-block:: python

   regions = profiler.identify_uncertainty_regions(
       X_test,
       threshold_percentile=90,
       return_regions=True,
   )
   
   high_unc = regions['high_uncertainty']
   print(f"High uncertainty at: {high_unc['points']}")
   
   # Check breakdown by region type
   print(regions['region_breakdown'])
   # {'EXTRAPOLATION': 40, 'INTERPOLATION': 5, ...}

Best Practices
--------------

1. **Always provide training data** for extrapolation detection
2. **Use validation data** to check calibration
3. **Monitor coefficient of variation**: CV > 5 suggests uneven uncertainty
4. **Check extrapolation ratio**: > 20% suggests test distribution mismatch

See Also
--------

- :class:`gpclarity.UncertaintyProfiler` for API details
- :doc:`complexity_analysis` for model capacity analysis