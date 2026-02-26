import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Deep Space Mission Design Toolkit'
copyright = '2026, Batuhan Akkova'
author = 'Batuhan Akkova'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
