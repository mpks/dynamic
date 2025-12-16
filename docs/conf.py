import sys
import os
import dynamic

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath('.'))
project = 'ccp4-ed'
copyright = '2025, Marko Petrovic'
author = 'Marko Petrovic'

numfig = True
numfig_format = {'figure': 'Fig. %s', }


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
}

mathjax3_config = {
        'chtml': {'displayAlign': 'left', 'displayIndent': '2em'},
        'tex': {'tags': 'ams'}   # Optional: Use AMS numbering style
}


templates_path = ['_templates']
exclude_patterns = []

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_context = {
  'display_github': True,
  'current_language': 'en',
  'current_version': "1.0",
  "github_user": "mpks",
}

html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_static_path = ['_static']
html_js_files = ["lightbox.js"]
html_css_files = ['custom.css', 'lightbox.css']
