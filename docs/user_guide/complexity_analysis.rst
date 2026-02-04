Complexity Analysis
===================

Is Your Model Too Simple or Too Complex?
----------------------------------------

GPClarity quantifies model complexity to detect overfitting and underfitting.

The Complexity Score
--------------------

GPClarity computes a composite score based on:

* **Kernel complexity**: Number of components and their interactions
* **Function roughness**: Inverse of lengthscales (higher = more wiggly)
* **Signal-to-noise ratio**: Balance between signal and noise variance
* **Effective degrees of freedom**: Model capacity relative to data

Basic Usage
-----------

.. code-block:: python

   from gpclarity import compute_complexity_score
   
   report = compute_complexity_score(model, X_train)
   
   print(f"Score: {report['log_score']:.2f} (log scale)")
   print(f"Category: {report['category']}")
   print(f"Interpretation: {report['interpretation']}")

Understanding Categories
------------------------

=======================  =============  ==================================
Category                   Log Score      Meaning
=======================  =============  ==================================
TOO_SIMPLE                 < -0.5         Severe underfitting risk
SIMPLE                     -0.5 to 0.5    Possible underfitting
MODERATE                   0.5 to 1.5     Well-balanced
COMPLEX                    1.5 to 2.5     Monitor for overfitting
TOO_COMPLEX                > 2.5          High overfitting risk
=======================  =============  ==================================

Detailed Metrics
----------------

.. code-block:: python

   report = compute_complexity_score(model, X, return_diagnostics=True)
   
   # Access detailed metrics
   metrics = report  # ComplexityMetrics object
   
   print(f"Effective DOF: {metrics.effective_degrees_of_freedom:.1f}")
   print(f"Capacity ratio: {metrics.capacity_ratio:.2%}")
   print(f"Roughness: {metrics.roughness_score:.3f}")
   print(f"SNR: {metrics.signal_noise_ratio:.2f}")

Component Breakdown
^^^^^^^^^^^^^^^^^^^

* **Effective DOF**: How many parameters worth of flexibility
* **Capacity ratio**: DOF / n_samples (> 80% is risky)
* **Roughness**: Function wiggliness (geometric mean of inverse lengthscales)
* **SNR**: Signal-to-noise ratio (< 0.1 is very noisy)

Risk Assessment
---------------

GPClarity automatically identifies risk factors:

.. code-block:: python

   report = compute_complexity_score(model, X)
   
   for risk in report['risk_factors']:
       print(f"⚠️ {risk}")
   
   for rec in report['recommendations']:
       print(f"💡 {rec}")

Common Scenarios
----------------

Scenario: "Model is too simple"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Log score < 0.5
   report = compute_complexity_score(model, X)
   # Recommendations:
   # - Add more kernel components
   # - Increase kernel flexibility
   # - Check if model captures all data trends

Scenario: "Model is too complex"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Log score > 1.5
   report = compute_complexity_score(model, X)
   # Recommendations:
   # - Simplify kernel structure
   # - Add strong priors on hyperparameters
   # - Use sparse approximation methods

Alternative Scoring Strategies
------------------------------

.. code-block:: python

   # Geometric: Based on eigenvalue spectrum
   report = compute_complexity_score(model, X, strategy='geometric')
   
   # Bayesian: Based on log-likelihood curvature
   report = compute_complexity_score(model, X, strategy='bayesian')

Quick Check
-----------

For rapid assessment:

.. code-block:: python

   from gpclarity import quick_complexity_check
   
   print(quick_complexity_check(model, X))
   # > MODERATE: Well-balanced complexity (log-score=0.85)

See Also
--------

- :class:`gpclarity.ComplexityAnalyzer` for advanced usage
- :doc:`kernel_interpretation` for understanding kernel complexity