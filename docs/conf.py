# docs/conf.py
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import gpclarity

project = 'GPClarity'
release = gpclarity.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

master_doc = 'index'
exclude_patterns = ['_build']
html_theme = 'sphinx_rtd_theme'
