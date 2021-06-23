from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizerFast

from unmasked import configs


def holistic_score_model_on_paradigm(model: RobertaForMaskedLM,
                                     tokenizer: RobertaTokenizerFast,
                                     path_paradigm: Path,
                                     lower_case: bool = False,  # necessary for Roberta-base trained on lower-cased data
                                     ) -> List[float]:
    """
    probe a model on a single paradigm.
    """

    # load sentences
    sentences_ = path_paradigm.read_text().split('\n')
    assert len(sentences_) % 2 == 0, len(sentences_)  # must be even
    assert sentences_[-1]

    # lowercase  (do this for fairseq models trained on custom lower-cased data)
    if lower_case:
        sentences = [s.lower() for s in sentences_]
    else:
        sentences = sentences_

    # compute scores
    cross_entropies = []
    with torch.no_grad():

        loss_fct = CrossEntropyLoss(reduction='none')

        for start in range(0, len(sentences), configs.Scoring.batch_size):
            # batch sentences
            end = min(len(sentences), start + configs.Scoring.batch_size)
            x = tokenizer(sentences[start:end],
                          padding='longest',
                          max_length=128,
                          return_tensors='pt')

            # dev notes:
            # this tokenizer works as expected: special tokens, punctuation, attention mask, tokenization.
            # cross-entropies for roberta-base are the same as those produced by BabyBERta probing logic

            # get loss
            output = model(**{k: v.to('cuda') for k, v in x.items()})
            logits_3d = output['logits']
            logits_for_all_words = logits_3d.permute(0, 2, 1)
            labels = x['input_ids'].cuda()
            loss = loss_fct(logits_for_all_words,  # need to be [batch size, vocab size, seq length]
                            labels,  # need to be [batch size, seq length]
                            )

            # compute avg cross entropy per sentence
            # to do so, we must exclude loss for padding symbols, using attention_mask
            cross_entropies += [loss_i[np.where(row_mask)[0]].mean().item()
                                for loss_i, row_mask in zip(loss, x['attention_mask'].numpy())]

    if not cross_entropies:
        raise RuntimeError(f'Did not compute cross entropies.')

    assert len(cross_entropies) == len(sentences)

    return cross_entropies

