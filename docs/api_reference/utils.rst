utils module
============

.. automodule:: gpclarity.utils
   :members:
   :undoc-members:
   :show-inheritance:

Internal Utilities
------------------

These functions are used internally but may be useful for advanced users.

.. warning::
   These utilities are not part of the stable public API and may change
   without deprecation warnings.

Validation Functions
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   _validate_array
   _validate_kernel_matrix
   _cholesky_with_jitter

Parameter Extraction
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   _extract_param_value
   extract_kernel_params_flat