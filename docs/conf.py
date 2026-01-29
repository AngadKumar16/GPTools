# docs/conf.py
# Configuration file for the Sphinx documentation builder.
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Add parent directory to path for autodoc
sys.path.insert(0, os.path.abspath('..'))
import gpclarity

# -- Project information -----------------------------------------------------
project = 'GPTools'
copyright = '2024, Angad Kumar'
author = 'Angad Kumar'
release = gpclarity.__version__
version = gpclarity.__version__
# -- General configuration ---------------------------------------------------
extensions = [
    # Core Sphinx extensions
    'sphinx.ext.autodoc',        # Automatic API documentation
    'sphinx.ext.autosummary',    # Summary tables
    'sphinx.ext.viewcode',       # View source code
    'sphinx.ext.napoleon',       # Support Google/NumPy docstrings
    
    # Inter-project linking
    'sphinx.ext.intersphinx',    # Link to other projects' docs
    
    # Markdown support
    'myst_parser',               # Markdown parsing
    
    # Jupyter notebook support
    'nbsphinx',                  # Render Jupyter notebooks
    
    # Additional useful extensions
    'sphinx.ext.githubpages',    # Publish to GitHub Pages
    'sphinx_copybutton',         # Copy button for code blocks
]

# -- Extension settings ------------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
    'special-members': '__init__',
}

autosummary_generate = True  # Automatically generate stub pages

# Napoleon settings (for Google/NumPy docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

# MyST parser settings (for Markdown)
myst_enable_extensions = [
    "dollarmath",       # LaTeX math support ($...$)
    "amsmath",          # Extended LaTeX math environments
    "deflist",          # Definition lists
    "html_admonition",  # HTML admonitions
    "html_image",       # HTML image handling
]
myst_heading_anchors = 3  # Generate anchors for headings

# nbsphinx settings (for Jupyter notebooks)
nbsphinx_execute = 'never'  # Don't execute notebooks during doc build
nbsphinx_allow_errors = False  # Fail build if notebook has errors

# -- Template and exclude paths ----------------------------------------------
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', 'venv']
include_patterns = ['**/*.rst', '**/*.md', '**/*.ipynb']

# -- HTML output options -----------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme

html_static_path = ['_static']  # Custom CSS/JS directory

# Theme customization
html_css_files = [
    'custom.css',  # Custom styling
]

html_theme_options = {
    # Navigation
    'navigation_depth': 4,              # Maximum depth of navigation tree
    'collapse_navigation': False,       # Keep navigation expanded
    'sticky_navigation': True,         # Keep nav bar visible when scrolling
    
    # Display
    'display_version': True,           # Show version number
    'prev_next_buttons_location': 'bottom',  # Next/previous page buttons
    
    # Titles
    'titles_only': False,              # Show full page titles
    'includehidden': True,             # Include hidden TOctree entries
}

# -- Internationalization ----------------------------------------------------
language = 'en'

# -- Intersphinx mapping for cross-project links ---------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.9', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'GPy': ('https://gpy.readthedocs.io/en/deploy/', None),
}

# -- Custom variables for use in templates -----------------------------------
html_context = {
    'github_user': 'AngadKumar16', 
    'github_repo': 'gptools',
    'github_version': 'main',
}