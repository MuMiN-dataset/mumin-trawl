'''Train the verdict classifier'''

import pandas as pd
from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback,
                          DataCollatorWithPadding)
from datasets import Dataset, load_metric
from typing import Dict
import os


def train():
    '''Load in the data, and train the verdict classifier'''

    # Disable tokenizers parallelism
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Force one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Load in the annotations
    annotations = pd.read_csv('data/new_verdict_annotations.csv')
    annotations_test = pd.read_csv('data/verdict_annotations_test.csv')

    # Make lower-case copy of annotations
    annotations_lower = annotations.copy()
    annotations_lower.raw_verdict = (annotations_lower.raw_verdict
                                                      .str
                                                      .lower())

    # Make upper-case copy of annotations
    annotations_upper = annotations.copy()
    annotations_upper.raw_verdict = (annotations_upper.raw_verdict
                                                      .str
                                                      .upper())

    # Make capital-case copy of annotations
    annotations_cap = annotations.copy()
    annotations_cap.raw_verdict = (annotations_cap.raw_verdict
                                                  .str
                                                  .capitalize())

    # Make title-case copy of annotations
    annotations_title = annotations.copy()
    annotations_title.raw_verdict = (annotations_title.raw_verdict
                                                      .str
                                                      .title())

    # Concatenate all the annotations
    annotations = pd.concat((annotations_lower,
                             annotations_upper,
                             annotations_cap,
                             annotations_title), axis=0).sample(frac=1.0)


    # Make train dataset
    train_dict = dict(doc=annotations.raw_verdict,
                      orig_label=annotations.verdict)
    train = Dataset.from_dict(train_dict)

    # Make test dataset
    test_dict = dict(doc=annotations_test.raw_verdict,
                     orig_label=annotations_test.verdict)
    test = Dataset.from_dict(test_dict)

    # Set up model parameters
    label2id = {'factual': 0, 'misinformation': 1, 'other': 2}
    id2label = {id: label for label, id in label2id.items()}
    cache_dir = '/media/secure/dan/huggingface'
    params = dict(num_labels=3,
                  id2label=id2label,
                  label2id=label2id,
                  cache_dir=cache_dir)

    # Load the model and tokenizer
    model_id = 'xlm-roberta-base'
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               **params)

    def tokenise(examples: dict) -> dict:
        doc = examples['doc']
        return tokenizer(doc, truncation=True, padding=True, max_length=512)

    train = train.map(tokenise, batched=True)
    test = test.map(tokenise, batched=True)

    def create_numerical_labels(examples: dict) -> dict:
        examples['label'] = [label2id[lbl] for lbl in examples['orig_label']]
        return examples

    train = (train.map(create_numerical_labels, batched=True)
                  .remove_columns(['doc', 'orig_label']))
    test = (test.map(create_numerical_labels, batched=True)
                  .remove_columns(['doc', 'orig_label']))

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='/media/secure/dan/verdict-classifier',
        evaluation_strategy='steps',
        logging_strategy='steps',
        save_strategy='steps',
        eval_steps=2000,
        logging_steps=2000,
        save_steps=2000,
        report_to='tensorboard',
        save_total_limit=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=1000,
        warmup_steps=len(train) // 4,
        gradient_accumulation_steps=8,
        metric_for_best_model='f1_macro',
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_model_id='saattrupdan/verdict-classifier'
    )

    # Set up data collator for feeding the data into the model
    data_collator = DataCollatorWithPadding(tokenizer, padding='longest')

    # Set up early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)

    def compute_metrics(predictions_and_labels: tuple) -> Dict[str, float]:
        '''Compute the metrics needed for evaluation.

        Args:
            predictions_and_labels (pair of arrays):
                The first array contains the probability predictions and the
                second array contains the true labels.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the
                metric values as values.
        '''
        f1_metric = load_metric('f1')
        prec_metric = load_metric('precision')
        logits, labels = predictions_and_labels

        # Get one-dimensional predictions
        predictions = logits.argmax(axis=-1)

        # Initialise score dict
        all_scores = dict()

        # Compute the f1 scores
        results = f1_metric.compute(predictions=predictions,
                                       references=labels,
                                       average=None)['f1']
        all_scores['f1_macro'] = results.mean()
        all_scores['f1_misinformation'] = results[label2id['misinformation']]
        all_scores['f1_factual'] = results[label2id['factual']]
        all_scores['f1_other'] = results[label2id['other']]

        # Compute the f1 scores
        results = prec_metric.compute(predictions=predictions,
                                      references=labels,
                                      average=None)['precision']
        all_scores['prec_macro'] = results.mean()
        all_scores['prec_misinformation'] = results[label2id['misinformation']]
        all_scores['prec_factual'] = results[label2id['factual']]
        all_scores['prec_other'] = results[label2id['other']]

        return all_scores

    # Initialise the Trainer object
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train,
                      eval_dataset=test,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics,
                      callbacks=[early_stopping])

    trainer.train()
    trainer.evaluate(test)
    trainer.push_to_hub()
