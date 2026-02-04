"""Sphinx configuration for GPClarity documentation."""

import os
import sys

# Path setup
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'GPClarity'
copyright = '2024, Angad Kumar'
author = 'Angad Kumar'

# Version from package
try:
    from gpclarity import __version__
    version = __version__
    release = __version__
except ImportError:
    version = '0.1.0'
    release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The suffix of source filenames
source_suffix = '.rst'

# The master toctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# Favicon
# html_favicon = '_static/favicon.ico'

# Logo
# html_logo = '_static/logo.png'

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Autosummary
autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'gpclarity', 'GPClarity Documentation',
     [author], 1)
]

# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text
epub_uid = ''

# A list of files that should not be packed into the epub file
epub_exclude_files = ['search.html']

# -- RTD specific ------------------------------------------------------------

# On RTD, don't try to build PDF if it fails
if os.environ.get('READTHEDOCS', None) == 'True':
    # Reduce memory usage
    autodoc_member_order = 'bysource'
    
    # Mock heavy dependencies if needed for import
    try:
        import GPy
    except ImportError:
        from unittest.mock import MagicMock
        
        class Mock(MagicMock):
            @classmethod
            def __getattr__(cls, name):
                return MagicMock()
        
        MOCK_MODULES = ['GPy', 'GPy.kern', 'GPy.models', 'GPy.core']
        sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)