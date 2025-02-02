# HMNI (Hello my name is)
# Fuzzy Name Matching with Machine Learning
# Author: Christopher Thornton (christopher_thornton@outlook.com)
# 2020-2020
# MIT Licence

import os
import re
import heapq
import joblib
import unidecode
import numpy as np
import pandas as pd
import torch
from random import randint
import editdistance

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from fuzzywuzzy import fuzz

from collections import Counter
from hmni import syllable_tokenizer
from hmni import input_helpers_torch as input_helpers
from hmni import preprocess_torch as preprocess
from hmni.siamese_network_torch import SiameseLSTM
import tarfile

from abydos.phones import *
from abydos.phonetic import PSHPSoundexFirst, PSHPSoundexLast, Ainsworth
from abydos.distance import (IterativeSubString, BISIM, DiscountedLevenshtein, Prefix, LCSstr, MLIPNS, Strcmp95,
                             MRA, Editex, SAPS, FlexMetric, JaroWinkler, HigueraMico, Sift4, Eudex, ALINE, Covington,
                             PhoneticEditDistance)

import logging

import sys

sys.modules['syllable_tokenizer'] = syllable_tokenizer
sys.modules['input_helpers'] = input_helpers
sys.modules['preprocess'] = preprocess


# guard against uncomputable recursion with max name length
class CovingtonGuard(Covington):
    def dist(self, src, tar, max_length=11):
        src = src[:max_length]
        tar = tar[:max_length]
        normalizer = self._weights[5] * min(len(src), len(tar))
        if len(src) != len(tar):
            normalizer += self._weights[7]
        normalizer += self._weights[6] * abs(abs(len(src) - len(tar)) - 1)
        return self.dist_abs(src, tar) / normalizer


class Matcher:

    def __init__(self, model='latin', prefilter=True, allow_alt_surname=True, allow_initials=True,
                 allow_missing_components=True):

        # user-provided parameters
        self.model = model
        self.allow_alt_surname = allow_alt_surname
        self.allow_initials = allow_initials
        self.allow_missing_components = allow_missing_components
        self.prefilter = prefilter
        if self.prefilter:
            self.refined_soundex = {
                'b': 1, 'p': 1,
                'f': 2, 'v': 2,
                'c': 3, 'k': 3, 's': 3,
                'g': 4, 'j': 4,
                'q': 5, 'x': 5, 'z': 5,
                'd': 6, 't': 6,
                'l': 7,
                'm': 8, 'n': 8,
                'r': 9
            }

        # verify user-supplied class arguments
        model_dir = self.validate_parameters()

        self.impH = input_helpers.InputHelper()
        # Phonetic Encoder
        self.pe = Ainsworth()
        # Soundex Firstname Algorithm
        self.pshp_soundex_first = PSHPSoundexFirst()
        # Soundex Lastname Algorithm
        self.pshp_soundex_last = PSHPSoundexLast()

        # Update feature names to match latest implementation
        self.feature_names = [
            'partial_ratio', 'token_sort', 'tkn_set', 'ipa_sim',
            'soundex_match', 'iterativesubstring', 'bisim',
            'discountedlevenshtein', 'prefix', 'lcsstr', 'mlipns',
            'strcmp95', 'mra', 'editex', 'saps', 'flexmetric',
            'jaro', 'higueramico', 'sift4', 'eudex', 'aline', 'covington',
            'phoneticeditdistance'
        ]

        # String Distance algorithms
        self.algos = [IterativeSubString(), BISIM(), DiscountedLevenshtein(), Prefix(), LCSstr(), MLIPNS(),
                      Strcmp95(), MRA(), Editex(), SAPS(), FlexMetric(), JaroWinkler(mode='Jaro'), HigueraMico(),
                      Sift4(), Eudex(), ALINE(), CovingtonGuard(), PhoneticEditDistance()]

        # Load models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Base model (Random Forest)
        self.baseModel = joblib.load(os.path.join(model_dir, 'base_model.pkl'))

        # Load vocabulary
        self.vocab = preprocess.VocabularyProcessor.load(os.path.join(model_dir, 'vocab_siamese.pkl'))

        # Initialize and load Siamese model
        self.siamese_model = SiameseLSTM(
            vocab_size=len(self.vocab.vocabulary_),
            embedding_size=300,
            hidden_units=50,
            n_layers=2,
            dropout=0.2
        ).to(self.device)

        # Load model with new checkpoint format
        checkpoint = torch.load(os.path.join(model_dir, 'siamese_model_final.pt'), map_location=self.device,
                                weights_only=True)
        self.siamese_model.load_state_dict(checkpoint['model_state_dict'])
        self.siamese_model.eval()

        # Meta model
        self.metaModel = joblib.load(os.path.join(model_dir, 'meta.pkl'))

        # Batch size for predictions
        self.batch_size = 64 if self.device.type == 'cpu' else 512

        # seen names (mapping dict from raw name to processed name)
        self.seen_names = {}
        # seen pairs (mapping dict from name pair tuple to similarity)
        self.seen_pairs = {}
        # user scores (mapping dict from name pair tuple to similarity)
        self.user_scores = {}

    def validate_parameters(self):
        # extract model tarball into directory if doesnt exist
        model_dir = os.path.join(os.path.dirname(__file__), "models", self.model)
        if not os.path.exists(model_dir):
            tar = tarfile.open(os.path.join(os.path.dirname(__file__), "models", self.model + ".tar.gz"), "r:gz")
            tar.extractall(os.path.join(os.path.dirname(__file__), "models"))
            tar.close()
        return model_dir

    def output_sim(self, sim, prob, threshold):
        if prob:
            return round(sim, 4)
        return 1 if sim >= threshold else 0

    def assign_similarity(self, name_a, name_b, score):
        if not (isinstance(name_a, str) and isinstance(name_b, str)):
            raise TypeError('Only strings are supported in add_score method')
        if score < 0 or score > 1:
            raise ValueError('Score must be a number between 0 and 1 (inclusive)')
        pair = tuple(sorted((name_a.lower().strip(), name_b.lower().strip()),
                            key=lambda item: (-len(item), item)))
        self.user_scores[hash(pair)] = score

    def similarity(self, name_a, name_b, prob=True, threshold=0.5, surname_first=False):
        """Calculate similarity between two names"""
        # input validation
        if not (isinstance(name_a, str) and isinstance(name_b, str)):
            raise TypeError('Only string comparison is supported in similarity method')

        if len(self.user_scores) != 0:
            # return user score if match
            pair = tuple(sorted((name_a.lower().strip(), name_b.lower().strip()),
                                key=lambda item: (-len(item), item)))
            score = self.seen_set(pair, self.user_scores)
            if score is not None:
                return self.output_sim(score, prob=prob, threshold=threshold)

        # empty or single character string returns 0
        if len(name_a) < 2 or len(name_b) < 2:
            return 0

        # exact match returns 1
        if name_a == name_b:
            return 1

        # preprocess names
        name_a = self.preprocess(name_a)
        name_b = self.preprocess(name_b)

        # empty or single character string returns 0
        if len(name_a) == 0 or len(name_b) == 0:
            return 0

        # check for missing name components
        missing_component = False
        one_component = False
        if len(name_a) == 1 and len(name_b) == 1:
            one_component = True
        elif len(name_a) == 1 or len(name_b) == 1:
            if not self.allow_missing_components:
                return 0
            missing_component = True

        if surname_first:
            fname_a, lname_a, fname_b, lname_b = name_a[-1], name_a[0], name_b[-1], name_b[0]
        else:
            fname_a, lname_a, fname_b, lname_b = name_a[0], name_a[-1], name_b[0], name_b[-1]

        # check for initials in first and lastnames
        if not self.allow_initials and any(len(x) == 1 for x in [fname_a, lname_a, fname_b, lname_b]):
            return 0

        # lastname conditions
        initial_lname = False
        if len(lname_a) == 1 or len(lname_b) == 1:
            if lname_a[0] != lname_b[0] and not missing_component:
                return 0
            initial_lname = True
        elif not one_component:
            if self.allow_alt_surname:
                if self.pshp_soundex_last.encode(lname_a) != self.pshp_soundex_last.encode(lname_b):
                    if not missing_component:
                        return 0
                elif missing_component:
                    return self.output_sim(0.5, prob=prob, threshold=threshold)
            elif lname_a != lname_b and not missing_component:
                return 0
            elif missing_component:
                return self.output_sim(0.5, prob=prob, threshold=threshold)

        # check initial match in firstname
        if len(fname_a) == 1 or len(fname_b) == 1:
            if fname_a[0] == fname_b[0]:
                return self.output_sim(0.5, prob=prob, threshold=threshold)
            return 0

        # check if firstname is same
        if fname_a == fname_b:
            if not missing_component and not initial_lname:
                return 1
            return self.output_sim(0.5, prob=prob, threshold=threshold)

        # sort pair to normalize
        pair = tuple(sorted((fname_a, fname_b), key=lambda item: (-len(item), item)))

        # prefilter candidates using heuristics on firstname
        if self.prefilter and not missing_component and pair[0][0] != pair[1][0]:
            encoded1 = set(self.refined_soundex.get(c) for c in set(pair[0][1:]))
            encoded2 = set(self.refined_soundex.get(c) for c in set(pair[1][1:]))
            encoded1.discard(None)
            encoded2.discard(None)
            if encoded1.isdisjoint(encoded2):
                return 0

        # return pair score if seen
        seen = self.seen_set(pair, self.seen_pairs)
        if seen is not None:
            if initial_lname:
                seen = min(0.5, seen)
            return self.output_sim(seen, prob=prob, threshold=threshold)

        # generate features for base-level model
        # features = self.featurize(pair)
        # make inference on meta model
        sim = self.meta_inf(pair[0], pair[1])

        if not missing_component:
            # add pair score to the seen dictionary
            self.seen_pairs[hash(pair)] = sim

        if initial_lname:
            sim = min(0.5, sim)

        return self.output_sim(sim, prob=prob, threshold=threshold)

    def fuzzymerge(self, df1, df2, how='inner', on=None, left_on=None, right_on=None, indicator=False,
                   limit=1, threshold=0.5, allow_exact_matches=True, surname_first=False):
        # parameter validation
        if not isinstance(df1, pd.DataFrame):
            df1 = pd.DataFrame(df1)
        if not isinstance(df2, pd.DataFrame):
            df2 = pd.DataFrame(df2)

        if not (0 < threshold < 1):
            raise ValueError('threshold must be decimal number between 0 and 1 (given = {})'.format(threshold))
        if how.lower() == 'right':
            df1, df2 = df2, df1
            left_on, right_on = right_on, left_on
            how = 'left'

        if on is None:
            k1, k2 = left_on, right_on
        else:
            k1, k2 = on, on
            right_on = on

        key = 'key'
        # if name key in columns - generate random integer
        while key in df1.columns:
            key = str(randint(1, 1000000))

        df1[key] = df1[k1].apply(
            lambda x: self.get_top_matches(x, df2[k2], limit=limit, thresh=threshold,
                                           exact=allow_exact_matches, surname_first=surname_first))
        df1 = df1.explode(key)
        df1[key] = df1[key].str.get(0)
        df1 = df1.merge(df2, how=how, left_on=key, right_on=right_on, indicator=indicator)
        del df1[key]
        return df1

    def dedupe(self, names, threshold=0.5, keep='longest', replace=False, reverse=True, surname_first=False, limit=3):
        # parameter validation
        if keep not in ('longest', 'frequent'):
            raise ValueError(
                'invalid arguement {} for parameter \'keep\', use one of -- longest, frequent, alpha'.format(keep))

        if keep == 'frequent':
            # make frequency counter
            count = Counter(names)

        if not replace:
            # early filtering of dupes by converting to set
            seen = set()
            seen_add = seen.add
            names = [x for x in names if not (x in seen or seen_add(x))]

        results = []
        for item in names:
            if item in results and replace is False:
                pass
            # find fuzzy matches
            matches = self.get_top_matches(item, names, limit=limit, thresh=threshold,
                                           exact=True, surname_first=surname_first)
            # no duplicates found
            if len(matches) == 0:
                results.append(item)

            else:
                # sort matches
                if keep == 'longest':
                    # sort by longest to shortest
                    matches = sorted(matches, key=lambda x: len(x[0]), reverse=reverse)
                elif keep == 'frequent':
                    # sort by most frequent, then longest
                    matches = sorted(matches, key=lambda x: (count[x[0]], len(x[0])), reverse=reverse)
                else:
                    # sort alphabetically
                    matches = sorted(matches, key=lambda x: x[0], reverse=reverse)
                if not (replace is False and matches[0][0] in results):
                    results.append(matches[0][0])
        return results

    def sum_ipa(self, name_a, name_b):
        feat1 = ipa_to_features(self.pe.encode(name_a))
        feat2 = ipa_to_features(self.pe.encode(name_b))
        score = sum(cmp_features(f1, f2) for f1, f2 in zip(feat1, feat2)) / len(feat1)
        return score

    def preprocess(self, name):
        # lookup name
        seen = self.seen_set(name, self.seen_names)
        if seen is not None:
            return seen
        # chained processing steps
        processed_name = re.sub('[^a-zA-Z]+', '', unidecode.unidecode(name).lower().strip()) \
            .replace('\'s', '').replace('\'', '')
        processed_name = [x for x in re.split(r'\+', processed_name) if x != '']
        # add processed name to the seen dictionary
        self.seen_names[hash(name)] = processed_name
        return processed_name

    def featurize(self, pair):
        if len(pair) != 2:
            raise ValueError(
                'Length mismatch: Expected axis has 2 elements, new values have {} elements'.format(len(pair)))
        # syllable tokenize names
        syll_a = syllable_tokenizer.syllables(pair[0])
        syll_b = syllable_tokenizer.syllables(pair[1])

        # generate unique features
        features = np.zeros(23)  # 5 base features + 17 algorithm features
        features[0] = fuzz.partial_ratio(syll_a, syll_b)  # partial ratio
        features[1] = fuzz.token_sort_ratio(syll_a, syll_b)  # sort ratio
        features[2] = fuzz.token_set_ratio(syll_a, syll_b)  # set ratio
        features[3] = self.sum_ipa(pair[0], pair[1])  # sum IPA
        features[4] = 1 if self.pshp_soundex_first.encode(pair[0]) == self.pshp_soundex_first.encode(
            pair[1]) else 0  # PSHPSoundexFirst
        # generate remaining features
        for i, algo in enumerate(self.algos):
            features[i + 5] = algo.sim(pair[0], pair[1])
        return features

    def transform_names(self, pair):
        x1 = np.asarray(list(self.vocab.transform(np.asarray([pair[0]]))))
        x2 = np.asarray(list(self.vocab.transform(np.asarray([pair[1]]))))
        return torch.from_numpy(x1).to(self.device), torch.from_numpy(x2).to(self.device)

    def siamese_inf(self, pair):
        x1, x2 = self.transform_names(pair)

        with torch.no_grad():
            distance = self.siamese_model(x1, x2)
            sim = 1 - distance[0].item()  # Convert distance to similarity
        return sim

    def process_name(self, name):
        """Preprocess a name for feature calculation"""
        if not name:
            return ''
        # Convert to lowercase and remove special characters
        processed_name = name.lower().strip()
        # Remove any characters that aren't letters or whitespace
        processed_name = re.sub(r'[^a-z\s]', '', processed_name)
        # Split into tokens and join with space
        return ' '.join(processed_name.split())

    def get_base_features(self, name_a, name_b):
        """Calculate base features for a name pair"""
        # Process names
        name_a = self.process_name(name_a)
        name_b = self.process_name(name_b)

        features = np.zeros(len(self.feature_names))

        # Calculate basic features
        features[0] = fuzz.partial_ratio(name_a, name_b)  # partial_ratio
        features[1] = fuzz.token_sort_ratio(name_a, name_b)  # token_sort
        features[2] = fuzz.token_set_ratio(name_a, name_b)  # tkn_set

        # IPA similarity
        ipa_a = self.pe.encode(name_a)
        ipa_b = self.pe.encode(name_b)
        features[3] = 1.0 - (editdistance.eval(ipa_a, ipa_b) / max(len(ipa_a), len(ipa_b)))  # ipa_sim

        # Soundex match
        features[4] = 1.0 if self.pshp_soundex_first.encode(name_a) == self.pshp_soundex_first.encode(name_b) else 0.0

        # Calculate remaining algorithm features
        for i, algo in enumerate(self.algos):
            try:
                features[i + 5] = algo.sim(name_a, name_b)
            except:
                features[i + 5] = 0.0

        return features

    def base_model_inf(self, x):
        # get the positive class prediction from model
        y_pred = self.baseModel.predict_proba(x.reshape(1, -1))[0, 1]
        return y_pred

    def meta_inf(self, name_a, name_b):
        """Get meta model prediction for a single pair"""
        # Get base features
        base_features = self.get_base_features(name_a, name_b)

        # Get predictions from base and siamese models
        base_pred = self.base_model_inf(base_features)
        siamese_pred = self.siamese_inf((name_a, name_b))

        # Create meta features
        meta_features = np.zeros(6)  # Changed from 5 to 6 features
        meta_features[0] = base_pred
        meta_features[1] = siamese_pred
        meta_features[2] = base_features[2]  # tkn_set
        meta_features[3] = base_features[5]  # iterativesubstring
        meta_features[4] = base_features[11]  # strcmp95
        meta_features[5] = base_pred * siamese_pred  # interaction term

        # Get meta model prediction
        return self.metaModel.predict_proba(meta_features.reshape(1, -1))[0, 1]

    def seen_set(self, item, mapping):
        h = hash(item)
        if h in mapping:
            return mapping[h]

    def get_top_matches(self, name, choices, thresh=0.5, exact=True, limit=1, surname_first=False):
        sl = self.get_matches(name, choices, thresh, exact, surname_first=surname_first)
        return heapq.nlargest(limit, sl, key=lambda i: i[1]) if limit is not None else sorted(
            sl, key=lambda i: i[1], reverse=True)

    def get_matches(self, name, choices, score_cutoff=0.5, exact=True, surname_first=False):
        # catch generators without lengths
        if choices is None or len(choices) == 0:
            return

        exact = 2 if exact is True else 1
        for choice in choices:
            score = self.similarity(name, choice, surname_first=surname_first)
            if exact > score >= score_cutoff:
                yield choice, score

    def get_predictions_batch(self, names_a, names_b):
        """Get predictions for a batch of name pairs"""
        num_pairs = len(names_a)
        base_preds = []
        siamese_preds = []

        # Process in batches
        for i in range(0, num_pairs, self.batch_size):
            batch_end = min(i + self.batch_size, num_pairs)
            batch_names_a = names_a[i:batch_end]
            batch_names_b = names_b[i:batch_end]

            # Get base features for batch
            batch_features = np.array([
                self.get_base_features(name_a, name_b)
                for name_a, name_b in zip(batch_names_a, batch_names_b)
            ])

            # Get base model predictions
            batch_base_preds = self.baseModel.predict_proba(batch_features)[:, 1]
            base_preds.extend(batch_base_preds)

            # Transform names for Siamese model
            x1_batch = np.asarray(list(self.vocab.transform(np.asarray(batch_names_a))))
            x2_batch = np.asarray(list(self.vocab.transform(np.asarray(batch_names_b))))
            x1_batch = torch.from_numpy(x1_batch).to(self.device)
            x2_batch = torch.from_numpy(x2_batch).to(self.device)

            # Get Siamese predictions
            with torch.no_grad():
                distances = self.siamese_model(x1_batch, x2_batch)
                similarities = 1 - distances.cpu().numpy()
            siamese_preds.extend(similarities)

        return np.array(base_preds), np.array(siamese_preds)

    def _process_batch(self, batch_data):
        """Helper function to process a batch of names in parallel"""
        name_a, name_b, threshold, surname_first = batch_data
        return self.similarity(name_a, name_b, prob=True, threshold=threshold, surname_first=surname_first)

    def similarity_batch(self, names_a, names_b, prob=True, threshold=0.5, surname_first=False, n_jobs=-1):
        """Calculate similarity between two lists of names in parallel batches"""
        if not (isinstance(names_a, (list, np.ndarray)) and isinstance(names_b, (list, np.ndarray))):
            raise TypeError('Input must be lists or numpy arrays')
        if len(names_a) != len(names_b):
            raise ValueError('Length of input lists must match')
        if len(names_a) == 0:
            return []

        import multiprocessing as mp
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        # Prepare batches for parallel processing
        batch_data = [(name_a, name_b, threshold, surname_first)
                      for name_a, name_b in zip(names_a, names_b)]

        # Process in parallel
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(self._process_batch, batch_data)

        if not prob:
            results = [1 if x >= threshold else 0 for x in results]

        return [round(x, 4) for x in results]

