"""
Adapted from https://github.com/awslabs/mlm-scoring, by ph (Philip Huebner), in June, 2021.

Removed all code not directly related to mlm-scoring on BLiMP,
and added code to support huggingface Roberta model.

Note:
    this script is called with working directory set to BabyBERTa/blimp.
"""

from contextlib import contextmanager
import logging
import sys
import numpy as np
from typing import List
import random
import os
import mxnet as mx
from pathlib import Path

from transformers.models.roberta import RobertaTokenizerFast

from unmasked.mlm.loaders import Corpus
from unmasked.mlm.scorer import MLMScorerPT

SEED = 0


def setup_ctxs(gpu_str: str) -> List[mx.Context]:

    random.seed(SEED)
    np.random.seed(SEED)
    mx.random.seed(SEED)

    ids = [int(id) for id in gpu_str.split(',')]
    if len(ids) == 1 and ids[0] < 0:
        ctxs = [mx.cpu(0)]
    else:
        for id in ids:
            mx.random.seed(SEED, mx.gpu(id))
        ctxs = [mx.gpu(id) for id in ids]

    # Following GluonNLP's scripts/language_model/large_word_language_model.py
    # https://mxnet.incubator.apache.org/faq/env_var.html
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    os.environ['MXNET_CPU_PARALLEL_RAND_COPY'] = str(len(ctxs))
    os.environ['MXNET_CPU_WORKER_NTHREADS'] = str(len(ctxs))

    return ctxs


def mlm_score_model_on_paradigm(model,
                                tokenizer: RobertaTokenizerFast,
                                path_paradigm: Path,
                                lower_case: bool,
                                ) -> List[float]:

    ctxs = setup_ctxs(gpu_str='0')

    # Set scorer
    vocab = tokenizer.get_vocab()
    scorer = MLMScorerPT(model, vocab, tokenizer, ctxs=ctxs)

    # What data do we use?
    infile = path_paradigm.open('r')
    corpus = Corpus.from_file(infile, lower_case=lower_case)
    assert len(corpus) == 2000 or len(corpus) == 4000

    # === START SHARED COMPUTATION ===
    scores = scorer.score(corpus, ratio=1, split_size=500)

    return scores
