#!/usr/bin/env python

import argparse
from mlmap import load_data, hf_tokenize_data
from mlmap.transformer_utils import CustomTrainer, compute_metrics, CustomTrainingArguments, optuna_hp_space
from transformers import AutoModelForSequenceClassification
from optuna.storages import RDBStorage
from optuna.study import delete_study
from sklearn.model_selection import KFold
import os
from pathlib import Path
import numpy as np
import json

def pipeline_train():
    parser = argparse.ArgumentParser(description='Run the inner loop of a nested cross validation process')
    parser.add_argument('-o', type=int, dest='outer_splits', default=3)
    parser.add_argument('-i', type=int, dest='inner_splits', default=3)
    parser.add_argument('-m', type=str, dest='model_name',
                        required=True)
    parser.add_argument('-y', type=str, dest='y_prefix',
                       default='INCLUDE')
    parser.add_argument('-t', type=int, dest='n_trials', 
                        default=1)
    args = parser.parse_args()    

    # Define a random seed to ensure replicability
    SEED = 2023

    # Load the data
    df, targets, weights, binary = load_data(args.y_prefix, random_state=SEED)

    # Uncomment this if you want to try the pipeline on just a small sample of labels
    # df = df.sample(100).reset_index(drop=True)
    
    # Tokenize the data
    dataset = hf_tokenize_data(df, args.model_name)

    # Initialise some custom training arguments
    training_args = CustomTrainingArguments(
        output_dir='./results',
        save_steps=1e9,
        optim = 'adamw_torch',
        class_weights = weights # setting the weights to the weights we have
    )
    
    # We'll store optuna trial results in an sqlite database
    storage = RDBStorage(
        url="sqlite:///results/trials.db"
    )
    model_dir = args.model_name.replace('/','__')

    # Define our model initialisation function, passing our model path and number of labels 
    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            cache_dir="transformers",
            num_labels=len(targets) if len(targets) > 2 else 2,
            ignore_mismatched_sizes=True
        )

    outer_fold = KFold(args.outer_splits)
    for k, (outer_train_idx, test_idx) in enumerate(outer_fold.split(df.index)):

        # Make a name for this study - in this case the combination of model name, variable we are predicting and outer fold
        study_name = f"{model_dir}__{args.y_prefix}__{k}"
        
        # Assuming we are running this a second time for a reason, delete the old results
        try:
            delete_study(study_name=study_name, storage=storage)
        except KeyError as e:
            pass # study does not exist
            
        # We'll store some results we calculate ourselves in a json file 
        results_path = Path(f'results/{study_name}.json')

        trainer = CustomTrainer(
            model=None,
            args=training_args,
            model_init=model_init,
            compute_metrics=compute_metrics
        )
        # Our custom trainer has multiple train and eval datasets which we 
        # set up from the inner loop
        trainer.train_datasets = []
        trainer.eval_datasets = []

        inner_fold = KFold(args.inner_splits)
        for j, (inner_train_idx, inner_validation_idx) in enumerate(inner_fold.split(outer_train_idx)):
            inner_train = outer_train_idx[inner_train_idx]
            val = outer_train_idx[inner_validation_idx]
            trainer.train_datasets.append(dataset.select(inner_train))
            trainer.eval_datasets.append(dataset.select(val))

        # Search for the best set of hyperparameters, optimising the f1 score
        # across inner validation sets for each set of params
        best_trial = trainer.cv_hp_search_optuna(
            direction='maximize',
            backend='optuna',
            hp_space=optuna_hp_space,
            n_trials=args.n_trials,
            storage=storage,
            study_name=study_name,
            load_if_exists=True
        )  


        #####
        ## Now that we've found our best trial, we'll train a model with these parameters, and test it on the outer test set
        best_hyperparameters = best_trial.hyperparameters
        for key, val in best_hyperparameters.items():
            setattr(training_args, key, val)

        # Train on our outer loop, using the best params from our inner cv procedure
        outer_train_ds = dataset.select(outer_train_idx)
        test_ds = dataset.select(test_idx)
        trainer = CustomTrainer(
            model=None,
            args=training_args,
            train_dataset=outer_train_ds,
            eval_dataset=test_ds,
            model_init=model_init,
            compute_metrics=compute_metrics
        )
        trainer.train()
        # Evaluate on our outer test set (which has not yet been seen by any model)
        scores = trainer.evaluate()

        # Make predictions for our outer test dataset (redundant, we've already evaluated, but I want the predictions themselves)
        preds = trainer.predict_proba(test_ds, binary=binary)    
        np.save(f"results/predictions/{study_name}__outer_predictions", preds)
        np.save(f"results/predictions/{study_name}__outer_ids", df.loc[test_idx,"id"].values)

        # Save the parameters and the scores in a json file
        results = {
            'hyperparameters': best_hyperparameters,
            'scores': scores
        }
        with open(f'results/{study_name}.json', 'w') as f:
            json.dump(results, f)

    ########
    ## Final model
    # Once we have done this across all folds, we can train our final model

    study_name = f"{args.model_name.replace('/','__')}__{args.y_prefix}_final"
    results_path = Path(f'results/{study_name}.json')

    # Assuming we are running this a second time for a reason, delete the old results
    try:
        delete_study(study_name=study_name, storage=storage)
    except KeyError as e:
        pass # study does not exist

    # We set up a trainer with train, val datasets from each of our cv splits
    trainer = CustomTrainer(
        model=None,
        args=training_args,
        model_init=model_init,
        compute_metrics=compute_metrics
    )
    trainer.train_datasets = []
    trainer.eval_datasets = []

    for k, (outer_train_idx, test_idx) in enumerate(outer_fold.split(df.index)):
        trainer.train_datasets.append(dataset.select(outer_train_idx))
        trainer.eval_datasets.append(dataset.select(test_idx))

    # Optimise the hyperparameters, finding the set of parameters that performs
    # the best averaged across our outer test sets
    best_trial = trainer.cv_hp_search_optuna(
        direction='maximize',
        backend='optuna',
        hp_space=optuna_hp_space,
        n_trials=args.n_trials,
        storage=storage,
        study_name=study_name,
        load_if_exists=True
    )
    # Save the best performing model
    results = {
        'hyperparameters': best_trial.hyperparameters,
    }
    with open(f'results/{study_name}.json', 'w') as f:
        json.dump(results, f)

    # Use the best hyperparameters to train a model on the whole dataset
    for k, v in results["hyperparameters"].items():
        setattr(training_args, k, v)

    # We train our final model on the whole dataset
    trainer = CustomTrainer(
        model=None,
        args=training_args,
        train_dataset=dataset,
        model_init=model_init,
    )
    result = trainer.train()
    trainer.save_model(f"results/{study_name}_model")

if __name__ == '__main__':
    pipeline_train()