import sys
import os
import dynamic

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# on_rtd = os.environ.get("READTHEDOCS") == "True"

# if on_rtd:
autodoc_mock_imports = [
    "dials",
    "dials.model",
    "dials.array_family",
    "cctbx",
    "scitbx",
    "iotbx",
    "libtbx",
]

sys.path.insert(0, os.path.abspath('.'))
project = 'Dynamic'
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
    "sphinxcontrib.bibtex",
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
  "github_user": "mpks",
  "github_repo": "dynamic",
  "github_version": "master",
  "conf_py_path": "/docs/",
  'current_language': 'en',
  'current_version': "1.0",
}

html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_static_path = ['_static']
html_js_files = ["lightbox.js"]
html_css_files = ['custom.css', 'lightbox.css']
bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"

rst_epilog = """
.. _point_nets: https://web.stanford.edu/~rqi/pointnet/
.. _ccp4-ed: https://www.ccp4.ac.uk/ccp4-ed/
.. _python-docs: https://docs.python.org/3/
"""
