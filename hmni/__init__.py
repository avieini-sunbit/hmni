from . import input_helpers_torch as input_helpers, matcher, preprocess_torch as preprocess, siamese_network_torch as siamese_network, syllable_tokenizer

import os
import tarfile

# extract model tarball into directory if doesnt exist
models = ['latin']

for model in models:
    model_dir = os.path.join(os.path.dirname(__file__), "models", model)
    if not os.path.exists(model_dir):
        tar = tarfile.open(os.path.join(os.path.dirname(__file__), "models", model + ".tar.gz"), "r:gz")
        tar.extractall(os.path.join(os.path.dirname(__file__), "models"))
        tar.close()

Matcher = matcher.Matcher

__all__ = [
    'input_helpers',
    'matcher',
    'preprocess',
    'siamese_network',
    'syllable_tokenizer'
] 