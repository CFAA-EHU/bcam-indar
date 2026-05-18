# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import bcam.indar
import re

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
    'sphinx.ext.intersphinx',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
autosummary_imported_members = True

# Inherit docstrings from parent class methods during autodoc generation.
autodoc_inherit_docstrings = True

# Generate autosummary pages for class members listed by numpydoc,
# including inherited ones coming from sklearn base classes.
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = True

# Some inherited sklearn docstrings include references that only resolve
# inside the sklearn documentation project.
nitpick_ignore = [
    ('std:ref', 'metadata_routing'),
    ('std:term', 'meta-estimator'),
]

# Keep inherited sklearn API docs, but suppress cross-project reference noise.
suppress_warnings = [
    'ref.ref',
    'ref.term',
]

# Cross-reference objects from external documentation projects.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# Hide the '(Source code)' link shown by matplotlib plot directive blocks.
plot_html_show_source_link = False
# Keep only PNG artifacts in plot output links.
plot_formats = ['png']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_nefertiti'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = [
    ('custom.js', {'priority': 100}),
]

# -- Math rendering options --------------------------------------------------
mathjax4_config = {
    'tex': {
        'macros': {
            'im': [r'\text{Im}', 0],
            're': [r'\text{Re}', 0],
            # Norm macros: ||v||, with different sizes
            'norm': [r'\lVert #1 \rVert', 1],
            'bignorm': [r'\Bigl\lVert #1 \Bigr\rVert', 1],
            'Bignorm': [r'\biggl\lVert #1 \biggr\rVert', 1],
        }
    }
}

# -- Type aliasing for documentation -----------------------------------------
# Simplify type hints in signatures for Sphinx autodoc
autodoc_typehints = 'signature'
autodoc_type_aliases = {
    'ArrayLike': 'array-like',
}

# Use short names (e.g., ArrayLike instead of numpy.typing.ArrayLike).
autodoc_typehints_format = 'short'


def _clean_type_alias_display(_app, _what, _name, _obj, _options, signature, return_annotation):
    """Normalize ForwardRef-style alias rendering in autodoc signatures."""
    # Example: TypeAliasForwardRef('array-like') -> array-like
    pattern = r"TypeAliasForwardRef\((['\"])(.*?)\1\)"

    if signature:
        signature = re.sub(pattern, r'\2', signature)
    if return_annotation:
        return_annotation = re.sub(pattern, r'\2', return_annotation)

    return signature, return_annotation


def setup(app):
    app.connect('autodoc-process-signature', _clean_type_alias_display)
