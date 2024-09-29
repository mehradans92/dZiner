# Configuration file for the Sphinx documentation builder.

import os
import sys
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('../..'))

# Add the examples directory to the Python path
examples_dir = os.path.abspath(os.path.join('..', '..', 'examples'))
sys.path.insert(0, examples_dir)


#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'dZiner'
copyright = '2024, Mehrad Ansari'
author = ''
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

html_logo = "_static/logo_transparent.png"  
html_static_path = ["_static"]

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'nbsphinx',
    'sphinx_book_theme'
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_show_copyright = True

# Configure the theme
html_theme_options = {
    "repository_url": "https://github.com/mehradans92/dziner",
    "use_repository_button": True,
    "show_navbar_depth": 2,
}

html_css_files = ['custom.css']

# -- MyST-NB settings
nb_execution_mode = "off"
nb_execution_allow_errors = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'myst_nb',
}



myst_enable_extensions = ["colon_fence", "substitution", "linkify", "html_admonition", "html_image"]

master_doc = 'index'

pygments_style = 'sphinx'