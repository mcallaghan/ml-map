from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, HPSearchBackend, PredictionOutput
from datasets import Dataset
from torch.nn import BCEWithLogitsLoss, Sigmoid, Softmax
from torch import tensor
import pandas as pd
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

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

