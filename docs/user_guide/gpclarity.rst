gpclarity module
================

.. automodule:: gpclarity
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Public Functions
----------------

.. autosummary::
   :nosignatures:

   summarize_kernel
   interpret_lengthscale
   interpret_variance
   format_kernel_tree
   compute_complexity_score
   compute_roughness_score
   compute_noise_ratio
   count_kernel_components
   check_model_health
   extract_kernel_params_flat
   get_lengthscale
   get_noise_variance
   quick_complexity_check
   quick_uncertainty_check
   compare_uncertainty_profiles

Classes
-------

.. autosummary::
   :nosignatures:

   UncertaintyProfiler
   HyperparameterTracker
   DataInfluenceMap

Constants
---------

.. autodata:: AVAILABLE
   :annotation: = Dictionary of feature availability flags