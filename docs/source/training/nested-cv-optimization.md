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

# Optimizing parameters with nested cross-validation

Testing each set of parameters one by one is referred to as **Grid Search**. It is a *brute force* approach to optimization that is highly inefficient, and dependent on the parameters we put into the grid.

We can employ smarter [optimization methods](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf) to search a given hyperparameter space.

[Huggingface](https://huggingface.co/docs/transformers/hpo_train) supports you in doing this by interfacing with other optimization libraries. In this example we will look at how to do this with [optuna](https://optuna.org/), although several other libraries are available.

## Setting up a trial with optuna

First we need to define a parameter space to explore

```
def optuna_hp_space(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0, 0.3),
        'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [8, 16]),
        'use_class_weights': trial.suggest_categorical('use_class_weights', [0, 1]),
        'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 8)
    }
```


Then we need to give our trainer object an evaluation dataset, and define the metrics we want it to compute.

We can define the number of parameter combinations we want to try (which should be defined by our compute budget).

## Cross-validation with optuna

Previously, we would simply do this process for each fold, and then find the parameters with the best performance across the folds.

However, it is unlikely that we will see the same combinations of parameters across each fold, we therefore need to adjust our trial process.

We can stipulate that for each trial, the set of parameters is trained and tested on `k` splits of our training dataset. We can then ask optuna to optimize for the mean score across those splits. See {meth}`mlmap.transformer_utils.run_cv_hp_search_optuna`
