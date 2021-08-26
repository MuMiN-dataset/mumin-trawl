'''Train the verdict classifier'''

import pandas as pd
import torch.utils.data as D
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .dataset import VerdictDataset
from .model import VerdictClassifier
from ..utils import root_dir


def train():
    '''Load in the data, and train the verdict classifier'''
    # Load in the annotated date
    annotations = pd.read_csv('data/verdict_annotations.csv')

    # Make the raw verdicts lower case and drop duplicates
    annotations['raw_verdict_en'] = annotations.raw_verdict_en.str.lower()
    annotations.drop_duplicates(subset='raw_verdict_en', inplace=True,
                                ignore_index=True)

    # Load the tokeniser and tokenise the plaintext verdicts
    transformer = 'sentence-transformers/paraphrase-mpnet-base-v2'
    tokeniser = AutoTokenizer.from_pretrained(transformer)
    tokens = tokeniser(annotations.raw_verdict_en.tolist(),
                       padding=True,
                       truncation=True,
                       return_tensors='pt')

    # # Split the data in a stratified fashion
    # train_idxs, val_idxs = train_test_split(annotations.index.tolist(),
    #                                         stratify=annotations.verdict,
    #                                         random_state=4242)

    # # Get the train and val tokens
    # train_tokens = {key: val[train_idxs] for key, val in tokens.items()}
    # val_tokens = {key: val[val_idxs] for key, val in tokens.items()}

    # # Get the train and val labels
    # train_labels = annotations.loc[train_idxs, 'verdict']
    # val_labels = annotations.loc[val_idxs, 'verdict']

    # # Build the datasets
    # train_dataset = VerdictDataset(tokens=train_tokens, labels=train_labels)
    # val_dataset = VerdictDataset(tokens=val_tokens, labels=val_labels)

    # # Wrap the datasets in dataloaders for batching
    # train_dl = D.DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_dl = D.DataLoader(val_dataset, batch_size=8)

    dataset = VerdictDataset(tokens=tokens, labels=annotations.verdict)
    dl = D.DataLoader(dataset, batch_size=8, shuffle=True)

    # Load the model and train it
    clf = VerdictClassifier(transformer=transformer)
    tb_logger = TensorBoardLogger('tb_logs', name='verdict_classifier')
    model_checkpoint = ModelCheckpoint(dirpath='models', monitor='train_f1',
                                       mode='max')
    trainer = pl.Trainer(gpus=1,
                         logger=tb_logger,
                         callbacks=[model_checkpoint],
                         max_epochs=1000)
    trainer.fit(clf, dl) #train_dl, val_dl)
