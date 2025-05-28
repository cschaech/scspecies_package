# -- Path setup --------------------------------------------------------------
import os
import sys

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

autodoc_mock_imports = [
    "torch", "muon", "scipy", "tqdm", "h5py", "gseapy", "mygene"
]

# -- Project information -----------------------------------------------------
project = 'scspecies'
author = 'Clemens Schächter'
copyright = '2025, Clemens Schächter'
html_title       = "scspecies Documentation"
html_short_title = "scspecies"
release = '0.1.0'  

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',        # <- auto-generate summary tables
    'sphinx.ext.viewcode',           # <- add “[source]” links
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'nbsphinx',
    'jupyter_sphinx',
]

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
}

templates_path = ['_templates']
exclude_patterns = []

autodoc_class_signature = 'separated'

napoleon_custom_sections = [('Effects', 'params_style')] 

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']