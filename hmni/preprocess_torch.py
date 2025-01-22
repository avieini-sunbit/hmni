"""PyTorch wrapper for the original HMNI preprocessing code."""

import re
import torch
import pickle
import collections
from typing import List, Iterator, Optional, Dict
import pandas as pd

def tokenizer(iterator):
    """Character-level tokenizer generator."""
    for value in iterator:
        # Handle NaN and float values
        if pd.isna(value):
            yield []
        else:
            # Convert to string if needed and lowercase
            if not isinstance(value, str):
                value = str(value)
            value = value.lower()
            # Split into characters, ignoring spaces and special characters
            yield [c for c in value if c.isalnum()]

class CategoricalVocabulary(object):
    """Original CategoricalVocabulary implementation."""
    def __init__(self, unknown_token="<UNK>", support_reverse=True):
        self._unknown_token = unknown_token
        self._mapping = {unknown_token: 0}
        self._support_reverse = support_reverse
        if support_reverse:
            self._reverse_mapping = [unknown_token]
        self._freq = collections.defaultdict(int)
        self._freeze = False

    def __len__(self):
        return len(self._mapping)

    def freeze(self, freeze=True):
        self._freeze = freeze

    def get(self, category):
        if category not in self._mapping:
            if self._freeze:
                return 0
            self._mapping[category] = len(self._mapping)
            if self._support_reverse:
                self._reverse_mapping.append(category)
        return self._mapping[category]

    def add(self, category, count=1):
        category_id = self.get(category)
        if category_id <= 0:
            return
        self._freq[category] += count

    def trim(self, min_frequency, max_frequency=-1):
        # Sort by alphabet then reversed frequency
        self._freq = sorted(
            sorted(
                self._freq.items(),
                key=lambda x: (isinstance(x[0], str), x[0])),
            key=lambda x: x[1],
            reverse=True)

        self._mapping = {self._unknown_token: 0}
        if self._support_reverse:
            self._reverse_mapping = [self._unknown_token]

        idx = 1
        for category, count in self._freq:
            if 0 < max_frequency <= count:
                continue
            if count <= min_frequency:
                break
            self._mapping[category] = idx
            idx += 1
            if self._support_reverse:
                self._reverse_mapping.append(category)

        self._freq = dict(self._freq[:idx - 1])

    def reverse(self, class_id):
        if not self._support_reverse:
            raise ValueError("This vocabulary wasn't initialized with "
                         "support_reverse to support reverse() function.")
        return self._reverse_mapping[class_id]

class VocabularyProcessor(object):
    """Original VocabularyProcessor implementation with PyTorch support."""
    def __init__(self,
               max_document_length,
               min_frequency=0,
               vocabulary=None,
               tokenizer_fn=None):
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        if vocabulary:
            self.vocabulary_ = vocabulary
        else:
            self.vocabulary_ = CategoricalVocabulary()
        if tokenizer_fn:
            self._tokenizer = tokenizer_fn
        else:
            self._tokenizer = tokenizer

    def fit(self, raw_documents):
        for tokens in self._tokenizer(raw_documents):
            for token in tokens:
                self.vocabulary_.add(token)
        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)
        self.vocabulary_.freeze()
        return self

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Modified to return PyTorch tensor instead of numpy array."""
        sequences = []
        for tokens in self._tokenizer(raw_documents):
            word_ids = [0] * self.max_document_length
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            sequences.append(word_ids)
        return torch.tensor(sequences, dtype=torch.long)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f) 