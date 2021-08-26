'''Dataset used to train the verdict classifier'''

import torch.utils.data as D
from typing import Sequence
import torch


class VerdictDataset(D.Dataset):
    '''A dataset used to train the verdict classifier.

    Args:
        tokens (dict):
            The tokens, as outputted by a HuggingFace tokenizer object.
        labels (sequence of str):
            The labels, which can be 'factual', 'misinformation' or 'other'.
    '''
    def __init__(self, tokens: dict, labels: Sequence[str]):
        self.tokens = tokens

        # Convert the labels to integers
        self.label_names = ['factual', 'misinformation', 'other']
        self.labels = torch.LongTensor([self.label_names.index(label)
                                        for label in labels])

    def __getitem__(self, idx: int):
        item = {key: val[idx] for key, val in self.tokens.items()}
        return item, self.labels[idx]

    def __len__(self):
        return len(self.labels)
