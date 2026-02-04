API Reference
=============

.. module:: gpclarity

Public API
----------

High-level interpretability functions:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   gpclarity.summarize_kernel
   gpclarity.interpret_lengthscale
   gpclarity.interpret_variance
   gpclarity.format_kernel_tree
   gpclarity.compute_complexity_score
   gpclarity.compute_roughness_score
   gpclarity.compute_noise_ratio
   gpclarity.count_kernel_components
   gpclarity.check_model_health
   gpclarity.extract_kernel_params_flat
   gpclarity.get_lengthscale
   gpclarity.get_noise_variance

Core Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   gpclarity.UncertaintyProfiler
   gpclarity.HyperparameterTracker
   gpclarity.DataInfluenceMap

Module Reference
----------------

.. toctree::
   :maxdepth: 1

   gpclarity
   kernel_summary
   uncertainty_analysis
   model_complexity
   hyperparam_tracker
   data_influence
   exceptions
   utils