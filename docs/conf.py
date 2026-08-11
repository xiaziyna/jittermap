import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "jittermap"
author = "Jamila Taaki"
copyright = "2026, Jamila Taaki"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"
