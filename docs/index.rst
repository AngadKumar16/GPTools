GPClarity: Interpretability Toolkit
===================================

A Python library for Gaussian Process interpretability and diagnostics.

Installation
------------

.. code-block:: bash

   pip install gpclarity

Quick Start
-----------

.. code-block:: python

   import gpclarity
   import GPy
   import numpy as np

   # Train a GP model
   X = np.linspace(0, 10, 50).reshape(-1, 1)
   y = np.sin(X).flatten() + 0.1 * np.random.randn(50)
   
   kernel = GPy.kern.RBF(1) + GPy.kern.White(1)
   model = GPy.models.GPRegression(X, y[:, None], kernel)
   model.optimize()
   
   # Get insights
   summary = gpclarity.summarize_kernel(model)

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
EOF
