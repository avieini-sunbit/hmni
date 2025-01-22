import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import unidecode
import re
import random
from itertools import combinations
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split

from hmni import syllable_tokenizer
from abydos.distance import (IterativeSubString, BISIM, DiscountedLevenshtein, Prefix, LCSstr, MLIPNS, Strcmp95,
                           MRA, Editex, SAPS, FlexMetric, JaroWinkler, HigueraMico, Sift4, Eudex, Covington, 
                           PhoneticEditDistance, ALINE)
from abydos.phonetic import PSHPSoundexFirst, Ainsworth
from abydos.phones import ipa_to_features, cmp_features

# Set random seed
np.random.seed(42)
random.seed(30)

# Initialize phonetic algorithms
pshp_soundex_first = PSHPSoundexFirst()
pe = Ainsworth()

# Initialize distance algorithms
algos = [IterativeSubString(), BISIM(), DiscountedLevenshtein(), Prefix(), LCSstr(), MLIPNS(), Strcmp95(),
         MRA(), Editex(), SAPS(), FlexMetric(), JaroWinkler(mode='Jaro'), HigueraMico(), Sift4(), Eudex(),ALINE(),
         Covington(), PhoneticEditDistance() ]

algo_names = ['iterativesubstring', 'bisim', 'discountedlevenshtein', 'prefix', 'lcsstr', 'mlipns', 'strcmp95', 
              'mra', 'editex', 'saps', 'flexmetric', 'jaro', 'higueramico', 'sift4', 'eudex', 'aline',
              'covington', 'phoneticeditdistance']

def sum_ipa(name_a, name_b):
    """Calculate IPA features similarity score."""
    feat1 = ipa_to_features(pe.encode(name_a))
    feat2 = ipa_to_features(pe.encode(name_b))
    score = sum(cmp_features(f1, f2) for f1, f2 in zip(feat1, feat2))/len(feat1)
    return score

def featurize(df):
    """Extract features from name pairs."""
    if len(df.columns)==3:
        df.columns=['a', 'b', 'target']
    elif len(df.columns)==2:
        df.columns=['a', 'b']
    else:
        df = df.rename(columns={df.columns[0]: 'a', df.columns[1]: 'b' })
    
    print(f"Processing {len(df)} rows...")
    
    # Name normalization
    print("Processing names...")
    df['name_a'] = df.apply(lambda row: re.sub(
        '[^a-zA-Z]+', '', unidecode.unidecode(row['a']).lower().strip()), axis=1)
    df['name_b'] = df.apply(lambda row: re.sub(
        '[^a-zA-Z]+', '', unidecode.unidecode(row['b']).lower().strip()), axis=1)
    
    # Process syllables
    print("Processing syllables...")
    df['syll_a'] = df.apply(lambda row: syllable_tokenizer.syllables(row.name_a), axis=1)
    df['syll_b'] = df.apply(lambda row: syllable_tokenizer.syllables(row.name_b), axis=1)
    
    # Calculate fuzzy ratios
    print("Calculating fuzzy ratios...")
    df['partial'] = df.apply(lambda row: fuzz.partial_ratio(row.syll_a,row.syll_b), axis=1)
    df['tkn_sort'] = df.apply(lambda row: fuzz.token_sort_ratio(row.syll_a,row.syll_b), axis=1)
    df['tkn_set'] = df.apply(lambda row: fuzz.token_set_ratio(row.syll_a,row.syll_b), axis=1)
    
    # Calculate IPA and soundex
    print("Calculating IPA and soundex...")
    df['sum_ipa'] = df.apply(lambda row: sum_ipa(row.name_a, row.name_b), axis=1)
    df['pshp_soundex_first'] = df.apply(
        lambda row: 1 if pshp_soundex_first.encode(row.name_a)==pshp_soundex_first.encode(row.name_b) else 0, axis=1)
    
    # Calculate all algorithm similarities
    print("Calculating algorithm similarities...")
    for i, algo in enumerate(algos):
        print(f"Processing algorithm {i+1}/{len(algos)}: {algo_names[i]}")
        df[algo_names[i]] = df.apply(lambda row: algo.sim(row.name_a, row.name_b), axis=1)
    
    df.drop(['syll_a', 'syll_b'], axis=1, inplace=True)
    return df

def prepare_data(data_file='name_pairs.txt'):
    """Phase 1: Prepare data for both models."""
    print("\n=== Phase 1: Data Preparation ===")
    print("Loading data...")
    alt_names = pd.read_csv(data_file, sep=",", names=['name_a', 'name_b'], header=None)
    alt_names['target'] = 1

    # Generate negative class
    print("Generating negative samples...")


    all_names = alt_names.loc[:, 'name_a':'name_b'].values.tolist()
    unique_names = list(set([item for items in all_names for item in items]))
    alt_pairs = list(zip(alt_names.name_a, alt_names.name_b)) + list(zip(alt_names.name_b, alt_names.name_a))
    comb = list(combinations(unique_names, 2))
    non_alt = list(set(comb) - set(alt_pairs))
    non_alt = pd.DataFrame(random.choices(non_alt, k=70040), columns=['name_a', 'name_b'])
    non_alt['target'] = 0

    # Combine and featurize
    df = pd.concat([alt_names, non_alt])
    print("Featurizing data...")
    df = featurize(df.dropna())

    # Prepare for models
    y = df.target
    X = df.drop(['target'], axis=1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    
    # Save splits for both models
    print("\nSaving data splits...")
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_data() 