# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ML-assisted Maps'
copyright = '2023, Max Callaghan'
author = 'Max Callaghan'
release = '0.1'

import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_thebe'
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

autosectionlabel_prefix_document = True

source_suffix = ['.rst','.md']

templates_path = ['_templates']
exclude_patterns = []
#nb_execution_mode = "force"
nb_execution_mode = 'cache'
nb_execution_excludepatterns = ['*finetuning.md','*params.md','*data/*']
nb_execution_timeout = 60

thebe_config = {
    "repository_url": "https://github.com/mcallaghan/ml-map",
    "repository_branch": "master",
}

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/mcallaghan/ml-map",
    "repository_branch": "master",
    # "launch_buttons": {
    #     "binderhub_url": "https://mybinder.org",
    #     #"colab_url": "https://colab.research.google.com/",
    #     #"deepnote_url": "https://deepnote.com/",
    #     "notebook_interface": "jupyterlab",
    #     #"thebe": True,
    #     # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    # },
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    # "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
    "path_to_docs": "docs/source",
    # "logo": {
    #     "image_dark": "_static/logo-wide-dark.svg",
    #     # "text": html_title,  # Uncomment to try text with logo
    # },
    "icon_links": [
        {
            "name": "MCC APSIS",
            "url": "https://apsis.mcc-berlin.net",
            "icon": "_static/MCC_Logo_RZ_rgb.jpg",
            "type": "local",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/mcallaghan/ml-map",
            "icon": "fa-brands fa-github",
        },
        # {
        #     "name": "PyPI",
        #     "url": "https://pypi.org/project/sphinx-book-theme/",
        #     "icon": "https://img.shields.io/pypi/dw/sphinx-book-theme",
        #     "type": "url",
        # },
    ],
    # For testing
    # "use_fullscreen_button": False,
    # "home_page_in_toc": True,
    # "extra_footer": "<a href='https://google.com'>Test</a>",  # DEPRECATED KEY
    # "show_navbar_depth": 2,
    # Testing layout areas
    # "navbar_start": ["test.html"],
    # "navbar_center": ["test.html"],
    # "navbar_end": ["test.html"],
    # "navbar_persistent": ["test.html"],
    # "footer_start": ["test.html"],
    # "footer_end": ["test.html"]
}

panels_add_bootstrap_css = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = 'ML for Systematic Maps'
