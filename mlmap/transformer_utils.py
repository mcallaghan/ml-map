from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, HPSearchBackend, PredictionOutput
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.nn import BCEWithLogitsLoss, Sigmoid, Softmax
from torch import tensor, cuda
import numpy as np
from dataclasses import dataclass, field
from datasets import Dataset
from copy import copy
from sklearn.metrics import f1_score

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

def run_cv_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    """
    Do cross-validation and optimize on the mean across each validation split
    """
    import optuna

    def _objective(trial, checkpoint_dir=None, opt_metric="eval_f1"):
        checkpoint = None
        metrics = []
        trainer.objective = None
        for td, ed in zip(trainer.train_datasets, trainer.eval_datasets):# in range(len(trainer.train_datasets)):
            k_trainer = copy(trainer)
            k_trainer.train_dataset = td
            k_trainer.eval_dataset = ed
            result = k_trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            print_summary(result)
            metrics.append(k_trainer.evaluate()[opt_metric])
        trainer.objective = np.mean(metrics)
        return trainer.objective
    
    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if labels.ndim==1 and logits.ndim==2:
            logits = logits[:,1]
        else:
            labels = labels.float()
        criterion = BCEWithLogitsLoss()
        if self.args.use_class_weights:
            criterion.pos_weight = self.args.class_weights
        loss = criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
    def predict_proba(self, test_dataset: Dataset, binary: bool) -> PredictionOutput:
        logits = self.predict(test_dataset).predictions
        if not binary:
            activation = Sigmoid()
            y_pred = activation(tensor(logits)).numpy()
        else:
            activation = Softmax(dim=1)
            y_pred = activation(tensor(logits)).numpy()[:,1]
        return y_pred
        
    def cv_hp_search_optuna(
        self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: Optional[Union["str"]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,        
    ) -> BestRun:
        """
        Run hyperparameter search using CV
        """
        self.hp_space = hp_space
        self.hp_search_backend = HPSearchBackend.OPTUNA
        best_run = run_cv_hp_search_optuna(self, n_trials, direction, **kwargs)
        
        self.hp_search_backend = None
        return best_run
    
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    if labels.ndim>1:
        activation = Sigmoid()
        y_pred = activation(tensor(logits))
        y_pred_binary = np.where(y_pred>0.5,1,0)
        return {"f1": f1_score(labels, y_pred_binary, average="macro")}
    else:
        activation = Softmax(dim=1)
        y_pred = activation(tensor(logits))[:,1]
        y_pred_binary = np.where(y_pred>0.5,1,0)
        return {"f1": f1_score(labels, y_pred_binary)}

@dataclass
class CustomTrainingArguments(TrainingArguments):
    use_class_weights: bool = field(default=False, metadata={"help": "Whether to use class weights in loss function"})
    class_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": (
                "the weights for each class to be passed to the loss function"
            )
        },
    )
    
def optuna_hp_space(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0, 0.3),
        'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [8, 16]),
        'use_class_weights': trial.suggest_categorical('use_class_weights', [0, 1]),
        'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 8)
    }