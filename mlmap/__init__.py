from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, HPSearchBackend, PredictionOutput
from datasets import Dataset
from torch.nn import BCEWithLogitsLoss, Sigmoid, Softmax
from torch import tensor, cuda
import pandas as pd
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import re
import sqlite3

device = "cuda:0" if cuda.is_available() else "cpu"

def return_search(db, study_name):
    """
    Get the results of hyperparameter trials for a study
    """
    sql_query = f"""
    SELECT s.study_id, s.study_name, t.trial_id, t.number, tv.value, tp.param_name, tp.param_value
    FROM trials t
    INNER JOIN 
        studies s 
    ON t.study_id = s.study_id
        AND s.study_name LIKE "%{study_name}__%"
    INNER JOIN 
        trial_values tv
    ON t.trial_id = tv.trial_id
    INNER JOIN 
        trial_params as tp
    ON
        t.trial_id = tp.trial_id;
    """
    with sqlite3.connect(db) as con:
        cur = con.cursor()
        cur.execute(sql_query)
        names = list(map(lambda x: x[0], cur.description))
        res = [dict(zip(names, x)) for x in cur.fetchall()] 
    df = pd.DataFrame.from_dict(res)
    return df

def load_data(y_prefix, inclusion_var="INCLUDE", min_samples=15, random_state=2023):
    """
    Load the labelled data, find the columns we need, apply any transformations necessary, return weights
    """
    df = (pd.read_feather('data/labels.feather')
               .sort_values('id')
               .sample(frac=1, random_state=random_state)
               .reset_index(drop=True)
              )
    df["seen"] = 1 
      
    targets = [x for x in df.columns if re.match(f"^{y_prefix}",x)]
    targets = [x for x in targets if df[x].sum()>min_samples]
    df = df.dropna(subset=targets).reset_index(drop=True)
    df = df.dropna(subset=["title", 'abstract']).reset_index(drop=True)

    if y_prefix not in inclusion_var:
        df = df[df[inclusion_var]==1].reset_index(drop=True)

    if len(targets)==1:
        df['labels'] = df[targets[0]]
        binary=True
    else:
        df['labels'] = list(df[targets].values.astype(int))
        binary=False
        
    
    weights = df.shape[0] / df[targets].sum(axis=0)
    weights = tensor(weights).to(device)
    df["text"] = df["title"] + ".  " + df['abstract']
    df = df[["id","text","labels"]+targets] 
    return df, targets, weights, binary

def hf_tokenize_data(df, model_name):
    """
    Turn the data into a Huggingface dataset
    """
    d = {
        'text': df['text']
    }
    if 'labels' in df.columns:
        d['labels'] = df['labels']
    dataset = Dataset.from_dict(d)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=512
    )
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding='max_length',
            truncation=True
        ), batched=True
    )
    dataset.set_format("torch")
    # Remove the now redundant text column
    return dataset.remove_columns("text")

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
        if hasattr(self.args, 'use_class_weights'):
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

@dataclass
class CustomTrainingArguments(TrainingArguments):
    use_class_weights: bool = field(default=False, metadata={"help": "Whether to use class weights in loss\
 function"})
    class_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": (
                "the weights for each class to be passed to the loss function"
            )
        },
    )

