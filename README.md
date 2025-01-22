# HMNI

Fuzzy name matching with machine learning. Perform common fuzzy name matching tasks including similarity scoring, record linkage, deduplication and normalization.

HMNI is trained on an internationally-transliterated Latin firstname dataset, where precision is afforded priority.

| Model      | Accuracy | Precision | Recall | F1-Score |
| ---------- | -------- | --------- | ------ | -------- |
| HMNI-Latin | 0.9393   | 0.9255    | 0.7548 | 0.8315   |

## Installation

```bash
pip install hmni
```

## Quick Usage Guide

### Initialize a Matcher Object
```python
import hmni
matcher = hmni.Matcher(model='latin')
```

### Single Pair Similarity
```python
matcher.similarity('Alan', 'Al')
# 0.6838303319889133

matcher.similarity('Alan', 'Al', prob=False)
# 1

matcher.similarity('Alan Turing', 'Al Turing', surname_first=False)
# 0.6838303319889133
```

### Record Linkage
```python
import pandas as pd

df1 = pd.DataFrame({'name': ['Al', 'Mark', 'James', 'Harold']})
df2 = pd.DataFrame({'name': ['Mark', 'Alan', 'James', 'Harold']})

merged = matcher.fuzzymerge(df1, df2, how='left', on='name')
```

### Name Deduplication and Normalization
```python
names_list = ['Alan', 'Al', 'Al', 'James']

matcher.dedupe(names_list, keep='longest')
# ['Alan', 'James']

matcher.dedupe(names_list, keep='frequent')
# ['Al', 'James']

matcher.dedupe(names_list, keep='longest', replace=True)
# ['Alan', 'Alan', 'Alan', 'James']
```

## Requirements

- Python >=3.9
- numpy
- pandas
- torch
- joblib
- unidecode
- fuzzywuzzy
- editdistance
- abydos

## License

MIT 