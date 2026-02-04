Installation
============

Requirements
------------

GPClarity requires Python 3.8 or later. The core library has minimal dependencies, but full functionality requires GPy.

Basic Installation
------------------

For analysis and lightweight usage:

.. code-block:: bash

   pip install gpclarity

This installs the core interpretability tools without heavy scientific computing dependencies.

Full Installation
-----------------

For complete functionality including GP model training:

.. code-block:: bash

   pip install gpclarity[full]

This installs GPy and emukit dependencies.

Development Installation
------------------------

To install from source with development tools:

.. code-block:: bash

   git clone https://github.com/yourusername/gpclarity.git
   cd gpclarity
   pip install -e ".[dev]"

Dependencies
------------

Core dependencies:
- numpy >= 1.20
- scipy >= 1.7

Optional dependencies:
- GPy >= 1.10 (for GP modeling)
- emukit >= 0.4 (for advanced features)
- matplotlib >= 3.4 (for plotting)
- pandas >= 1.3 (for DataFrame export)

Verifying Installation
----------------------

.. code-block:: python

   import gpclarity
   print(gpclarity.__version__)
   print(f"Full features available: {gpclarity.AVAILABLE['full']}")