# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import setuptools_scm
sys.path.insert(0, os.path.abspath('..'))

project = 'AutoPDEx'
copyright = '2024, Tobias Bode'
author = 'Tobias Bode'
version = '1.1.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_rtd_dark_mode',
    'nbsphinx',
    'jupyter_sphinx',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

nbsphinx_allow_errors = True

mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

autosummary_generate = True
autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance']

master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_logo = '_static/logo.png'
html_favicon = ''
html_static_path = ['_static']
html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

# Add the link to the source code
html_context = {
    'display_github': True,  # Integrates with GitHub
    'github_user': 'BodeTobias',  # Your GitHub username
    'github_repo': 'AutoPDEx',  # Your repository name
    'github_version': 'main',  # The version of the repository (branch name)
    'conf_py_path': '/docs/',  # Path to your docs directory
}

def skip_member(app, what, name, obj, skip, options):
  if name == '__init__':
    return True
  return skip

# Custom CSS
def setup(app):
  app.add_css_file('custom.css')
  app.connect("autodoc-skip-member", skip_member)

default_dark_mode = False
