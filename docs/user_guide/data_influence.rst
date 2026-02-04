Data Influence Analysis
=======================

Which Data Points Matter?
-------------------------

GPClarity identifies influential observations and quantifies their impact on model uncertainty.

Basic Usage
-----------

.. code-block:: python

   from gpclarity import DataInfluenceMap
   
   influence = DataInfluenceMap(model)
   
   # Fast leverage scores (O(n³))
   result = influence.compute_influence_scores(X_train)
   scores = result.scores
   
   print(f"Most influential point: index {np.argmax(scores)}")

Leverage vs. Leave-One-Out
--------------------------

**Leverage Scores** (fast)
   * Approximate influence via hat matrix diagonal
   * O(n³) complexity
   * Good for initial screening

**Leave-One-Out** (precise)
   * Exact variance increase when removing each point
   * O(n⁴) complexity
   * Use for detailed analysis on small datasets

Leave-One-Out Analysis
----------------------

.. code-block:: python

   # Slower but more accurate
   loo_var, loo_err = influence.compute_loo_variance_increase(X_train, y_train)
   
   # loo_var: Increase in predictive variance without this point
   # loo_err: Absolute prediction error for this point

Influence Report
----------------

Comprehensive analysis with one call:

.. code-block:: python

   report = influence.get_influence_report(X_train, y_train)
   
   print(f"Mean influence: {report['influence_scores']['mean']:.3f}")
   print(f"High leverage points: {report['diagnostics']['high_leverage_points']}")
   
   # Most influential
   most = report['most_influential_point']
   print(f"Index {most['index']} at {most['location']} (score={most['score']:.3f})")

Visualization
-------------

.. code-block:: python

   influence.plot_influence(X_train, scores)
   
   # Shows:
   # - Data points sized by influence
   # - Color-coded by influence score
   # - Highlights high-leverage observations

Interpreting Influence
----------------------

High Influence Points
^^^^^^^^^^^^^^^^^^^^^

Points with high influence:
- Reduce uncertainty significantly when included
- Have high leverage on predictions
- May be outliers or in sparse regions

Actions:
- Verify data quality for high-influence points
- Check for errors or measurement issues
- Consider robust GPs if outliers are problematic

Low Influence Points
^^^^^^^^^^^^^^^^^^^^

Points with low influence:
- Don't affect model much when removed
- May be redundant (near other points)
- Could be removed for computational efficiency

Parallel Computation
--------------------

For large datasets, use parallel LOO:

.. code-block:: python

   loo_var, loo_err = influence.compute_loo_variance_increase(
       X_train, y_train,
       n_jobs=-1,  # Use all cores
       verbose=True,  # Show progress bar
   )

Best Practices
--------------

1. **Start with leverage scores** for screening
2. **Use LOO on subset** if n > 1000 (too slow for full dataset)
3. **Check high-influence points** for data errors
4. **Consider removing** zero-influence duplicates

See Also
--------

- :class:`gpclarity.DataInfluenceMap` for API details
- :class:`gpclarity.InfluenceResult` for result structure