"""
Sphinx configuration for GPDiagnostics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import gptools

project = 'GPDiagnostics'
copyright = '2024, Your Name'
author = 'Your Name'
release = gptools.__version__
version = gptools.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napole
