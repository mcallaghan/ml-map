BERT and transformer-based models
======

BERT (Bidirectional Encoder Representations from Transformers) (`Devlin et al. 2019 <https://arxiv.org/abs/1810.04805>`_) has changed how we machine learning with text

Put very simply `more detail here <https://jalammar.github.io/illustrated-bert/>`_, a large model is trained to predict missing words from billions of sentences.

.. figure:: ../images/bert-transfer-learning.png
   :alt: reStructuredText, the markup syntax

   (Source, `Alammar, J. The Illustrated Transformer <https://jalammar.github.io/illustrated-bert/>`_)

In doing so, the model learns (from how text is used by humans) how to represent words as **vectors**, with different vector representations for words depending on the context in which they occurred.

These useful vector representations mean that the **pretrained** model can be **fine-tuned** on custom data, to do a traditional classification task (as well as other tasks), such as "is this text about climate policy or not". If we show this model enough texts that are labelled as **About climate policy** and **not about climate policy**, then it should learn how to distinguish climate-policyness in new texts.
