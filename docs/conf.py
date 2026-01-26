"""
Sphinx configuration for GPDiagnostics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import gpdiagnostics

project = 'GPDiagnostics'
copyright = '2024, Your Name'
author = 'Your Name'
release = gpdiagnostics.__version__
version = gpdiagnostics.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napole
