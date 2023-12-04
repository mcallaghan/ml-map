---
title: Producing a Living Systematic Map Using Machine Learning
---

This tutorial takes you through the process of using machine learning to generate a systematic map of climate policy instruments literature, based on the one available [here](https://apsis.mcc-berlin.net/climate-policy-instruments-map/)

To run the code, it is recommended that you set up a virtual environment

```
python3 -m venv .venv
source .venv/bin/activate
```

Before you install python packages from the requirements file, you should install torch according to the instructions, paying attention to the architecture that suits your machine [here](https://pytorch.org/). If you have GPU resources with CUDA 12.1, you can simply `pip install torch`, for any other computational platform, you will want to adjust the --index-url according to the instructions.

Once torch is installed, you can install the remaining requirements with

```
pip install -r requirements
```

You should also install the code in this library via

```
pip install -e .
```

The tutorial is made using [myst-nb](https://myst-nb.readthedocs.io/en/latest/index.html). To rebuild the documentation, re-running all of the code yourself, you can run

```
sphinx-build -M html docs/source/ docs/build/
```

from the `docs` directory.

Alternatively, each page of the tutorial that contains code can be run as an .ipynb notebook, these are collected in `docs/jupyter_execute`

# Tutorial

```{toctree}
:maxdepth: 2
:caption: Tutorial

introduction/index
data/index
training/index
mapping/index

```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/api

```
