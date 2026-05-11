# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import bcam.indar

project = 'Indar'
copyright = '2026, BCAM'

version = bcam.indar.__version__
release = version

print(f"{project} (VERSION {version})")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
autosummary_imported_members = True

# Generate autosummary pages for class members listed by numpydoc,
# including inherited ones coming from sklearn base classes.
numpydoc_class_members_toctree = True
numpydoc_show_inherited_class_members = True

# Some inherited sklearn docstrings include references that only resolve
# inside the sklearn documentation project.
nitpick_ignore = [
    ('std:ref', 'metadata_routing'),
    ('std:term', 'meta-estimator'),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_nefertiti'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']
