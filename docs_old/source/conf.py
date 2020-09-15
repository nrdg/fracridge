# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'fracridge'
copyright = '2020, Kendrick Kay and Ariel Rokem'
author = 'Kendrick Kay and Ariel Rokem'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    ]

# Automatically generate stub pages for API
autoclass_content = "both"  # include both class docstring and __init__
autodoc_default_flags = ["members", "inherited-members"]
autosummary_generate = True

# Napoleon settings (other than default)
napoleon_google_docstring = False
napoleon_use_rtype = False

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../../examples",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    # To auto-generate example sections in the API
    "doc_module": ("fracridge",),
    # Auto-generated mini-galleries go here
    "backreferences_dir": "gen_api",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']