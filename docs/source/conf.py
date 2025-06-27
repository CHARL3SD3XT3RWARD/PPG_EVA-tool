# Configuration file for the Sphinx documentation builder.
#
import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath(r'C:\Users\akorn\Desktop\Charié\BA\final_version\PPG_EVA-tool'))  # Beispiel: zwei Ebenen nach oben
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PPG_Eva-tool'
copyright = '2025, A.Kornrumpf'
author = 'A.Kornrumpf'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
