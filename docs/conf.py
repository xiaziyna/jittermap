import os
import shutil
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
    "nbsphinx",
]

html_theme = "sphinx_rtd_theme"
autodoc_member_order = "bysource"

# Tutorial notebooks live in the top-level notebooks/ directory (committed
# with executed outputs); copy them into the docs source tree at build time
# so nbsphinx can render them without re-execution.
nbsphinx_execute = "never"
_here = os.path.dirname(os.path.abspath(__file__))
_nb_src = os.path.join(_here, "..", "notebooks")
_nb_dst = os.path.join(_here, "tutorials")
if os.path.isdir(_nb_src):
    os.makedirs(_nb_dst, exist_ok=True)
    for fname in os.listdir(_nb_src):
        if fname.endswith(".ipynb"):
            shutil.copyfile(os.path.join(_nb_src, fname),
                            os.path.join(_nb_dst, fname))

# README images referenced from the examples page
_img_src = os.path.join(_here, "..", "images")
_img_dst = os.path.join(_here, "_images_repo")
if os.path.isdir(_img_src):
    os.makedirs(_img_dst, exist_ok=True)
    for fname in os.listdir(_img_src):
        if fname.endswith((".png", ".gif")):
            shutil.copyfile(os.path.join(_img_src, fname),
                            os.path.join(_img_dst, fname))

exclude_patterns = ["_build"]
