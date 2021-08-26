'''Classifier to convert plaintext verdicts into binary values'''

from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from typing import List
import torch
import torchmetrics
import pandas as pd


class GMP(torch.nn.Module):
    '''A global max pooling module.

    Args:
        dim (int): The dimension at which to compute the maximum.
    '''
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)[0]


class VerdictClassifier(pl.LightningModule):
    '''A classifier to convert plaintext verdicts into binary values.

    Args:
        class_weights (list of floats):
            Weights given to the classes, where the classes are 'factual',
            'misinformation' and 'other'. Defaults to [1.0, 1.0, 1.0], meaning
            equal weight to all classes.
        transformer (str):
            The pretrained transformer to use to embed the verdicts. Defaults
            to 'google/electra-base-generator'.
    '''
    def __init__(self, class_weights: List[float] = [1., 1., 1.],
                 transformer: str = ('sentence-transformers/'
                                     'paraphrase-mpnet-base-v2')):
        super().__init__()
        self._tokeniser = AutoTokenizer.from_pretrained(transformer)
        self.transformer = AutoModel.from_pretrained(transformer)

        self.clf = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            GMP(dim=1),
            torch.nn.Linear(1024, 3)
        )

        weight = torch.FloatTensor(class_weights)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)
        self.f1_score = torchmetrics.F1(num_classes=3, average='macro')

        self._label_names = ['factual', 'misinformation', 'other']

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

    def forward(self, x):
        x = self.transformer(**x)
        x = x.last_hidden_state
        x = self.clf(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        self.log('train_loss', float(loss))

        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        self.f1_score(preds, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        self.log('val_loss', float(loss))

        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_f1', self.f1_score.compute())

    def predict(self, verdicts: List[str]) -> pd.DataFrame:
        '''Convert plaintext verdicts into labels.

        The three possible labels are 'factual', 'misinformation' and 'other'.

        Args:
            verdicts (list of str): The plaintext verdicts.

        Returns:
            Pandas DataFrame:
                Dataframe with columns 'raw_verdict', 'predicted_verdict' and
                'probability'
        '''
        self.eval()

        # Make all the verdicts lower case
        verdicts = [verdict.lower() for verdict in verdicts]

        # Tokenise the plaintext verdicts and convert them into tensors
        tokens = self._tokeniser(verdicts, padding=True, truncation=True,
                                 return_tensors='pt')

        # Get the logits, probabilities and predicted classes
        logits = self(tokens)
        probs, _ = torch.softmax(logits, dim=-1).max(dim=-1)
        probs = probs.detach()
        preds = logits.argmax(dim=-1)

        # Convert the classes into the names of the labels
        labels = [(verdicts[i], self._label_names[cls_idx], probs[i].item())
                  for i, cls_idx in enumerate(preds)]

        cols = ['raw_verdict', 'predicted_verdict', 'probability']
        return pd.DataFrame.from_records(labels, columns=cols)
