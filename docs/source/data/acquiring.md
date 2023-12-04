---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.8.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Acquiring Data

Data comes from [OpenAlex](https://openalex.org/).

There are more details on how to access OpenAlex data using the API available [here](https://github.com/mcallaghan/NLP-climate-science-tutorial-CCAI/blob/main/A_obtaining_data.ipynb)

In our case, for full flexibility, we download the fortnightly snapshot, and make this searchable using [Solr](https://solr.apache.org/).

We extracted climate-relevant articles using the following query:

```
((CO2 OR "carbon dioxide" OR methane OR CH4 OR "carbon cycle" OR "carbon cycles" OR "carbon cycling" OR "carbon budget" OR "carbon budgets" OR "carbon flux" OR "carbon fluxes" OR "carbon mitigation") AND (climat*)) OR
(("carbon cycle" OR "carbon cycles" OR "carbon cycling" OR "carbon budget" OR "carbon budgets" OR "carbon flux" OR "carbon fluxes" OR "carbon mitigation") AND (atmospher*)) OR

("carbon emission" OR "carbon emissions" OR "sequestration of carbon" OR "sequestered carbon" OR "sequestering carbon" OR "sequestration of CO2" OR "sequestered CO2" OR "sequestering CO2" OR "carbon tax" OR "carbon taxes" OR "CO2 abatement" OR "CO2 capture" OR "CO2 storage" OR "CO2 sequestration" OR "CO2 sink" OR "CO2 sinks" OR "anthropogenic carbon" OR "capture of carbon dioxide" OR "capture of CO2" OR "climate variability" OR "climatic variability" OR "climate dynamics" OR "change in climate" OR "change in climatic" OR "climate proxies" OR "climate proxy" OR "climate sensitivity" OR "climate shift" OR "climatic shift" OR "coupled ocean-climate" OR "early climate" OR "future climate" OR "past climate" OR "shifting climate" OR "shifting climatic" OR "shift in climate" OR "shift in climatic") OR

("atmospheric carbon dioxide" OR "atmospheric CH4" OR "atmospheric CO2" OR "atmospheric methane" OR "atmospheric N2O" OR "atmospheric nitrous oxide" OR "carbon dioxide emission" "carbon dioxide emissions" OR "carbon sink" OR "carbon sinks" OR "CH4 emission" OR "CH4 emissions" OR "climate policies" OR "climate policy" OR "CO2 emissions" OR "CO2 emission" OR dendroclimatology OR dendroclimatological OR ("emission of carbon dioxide" NOT nanotube*) OR ("emissions of carbon dioxide" NOT nanotube*) OR "emission of CH4" OR "emissions of CH4" OR "emission of CO2" OR "emissions of CO2" OR "emission of methane" OR "emissions of methane" OR "emission of N2O" OR "emissions of N20" OR "emission of nitrous oxide" OR "emissions of nitrous oxide" OR "historical climate" OR "historical climatic" OR IPCC OR "Intergovernmental Panel on Climate Change" OR "methane emission" OR "methane emissions" OR "N2O emission" OR "N20 emissions" OR "nitrous oxide emission" OR "nitrous oxide emissions") OR

("climate change" OR "climatic change" OR "climate changes" OR "climatic changes" OR "global warming" OR "greenhouse effect" OR "greenhouse gas" OR "greenhouse gases" OR "Kyoto Protocol" OR "warming climate" OR "warming climatic" OR "cap and trade" OR "carbon capture" OR "carbon footprint" OR "carbon footprints" OR "carbon neutral" OR "carbon neutrality" OR "carbon offset" OR "carbon sequestration" OR "carbon storage" OR "carbon trading" OR "carbon trade" OR "changing climate" OR "changing climatic" OR "climate warming" OR "climatic warming")

```

In this example tutorial, we make a small sample of our search results available in the data folder.

```{code-cell} ipython3
:tags: [hide-cell, thebe-init]

import os
os.chdir('../../../')
```

These are stored in a [.feather](https://arrow.apache.org/docs/python/feather.html) file

```{code-cell} ipython3
:tags: [thebe-init]

import pandas as pd
df = pd.read_feather('data/documents.feather')
print(df.shape)
df.head()
```
