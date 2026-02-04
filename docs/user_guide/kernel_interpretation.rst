Kernel Interpretation
=====================

Understanding Kernel Hyperparameters
------------------------------------

The kernel (covariance function) is the heart of a Gaussian Process. GPClarity translates cryptic hyperparameters into actionable insights.

Lengthscale Interpretation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The lengthscale ($\ell$) controls how quickly the function can change:

.. math::

   k(x, x') = \sigma^2 \exp\left(-\frac{(x-x')^2}{2\ell^2}\right)

GPClarity categorizes lengthscales:

* **Rapid variation** ($\ell < 0.5$): Function can change quickly, fits local patterns, risk of overfitting
* **Moderate** ($0.5 \leq \ell \leq 2.0$): Balanced flexibility for most applications  
* **Smooth trends** ($\ell > 2.0$): Slow variations only, may underfit complex patterns

Example:
   >>> from gpclarity import interpret_lengthscale
   >>> interpret_lengthscale(0.1)
   'Rapid variation (high frequency, 0.10)'
   >>> interpret_lengthscale(5.0)
   'Smooth trends (low frequency, 5.00)'

Variance Interpretation
^^^^^^^^^^^^^^^^^^^^^^^

Signal variance ($\sigma^2$) determines output magnitude:

* **Very low** ($< 0.01$): Function stays near zero, possibly underconfident
* **Moderate** ($0.01$ to $10$): Typical range for standardized data
* **High** ($> 10$): Large output scale, check for unstandardized data

Composite Kernels
^^^^^^^^^^^^^^^^^

Real-world models often combine kernels:

.. code-block:: python

   # RBF for smooth trend + Periodic for seasonality + White for noise
   kernel = GPy.kern.RBF(1) + GPy.kern.PeriodicExponential(1) + GPy.kern.White(1)
   
   summary = gpclarity.summarize_kernel(model)
   # Shows contribution of each component

Each component is interpreted separately with context-aware messaging.

ARD (Automatic Relevance Determination)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-dimensional inputs with separate lengthscales per dimension:

.. code-block:: python

   kernel = GPy.kern.RBF(input_dim=3, ARD=True)
   # Each dimension gets its own lengthscale
   
   summary = gpclarity.summarize_kernel(model)
   # Shows which dimensions are important (small lengthscale = important)

Interpreting Results
--------------------

GPClarity provides specific recommendations:

* **"Consider more expressive kernel"**: Current kernel too simple for data patterns
* **"Monitor for overfitting"**: High complexity, ensure validation performance holds
* **"Well-balanced flexibility"**: Lengthscale appropriate for typical data scales

See Also
--------

* :doc:`complexity_analysis` for quantifying overall model complexity
* :class:`gpclarity.InterpretationConfig` for customizing thresholds