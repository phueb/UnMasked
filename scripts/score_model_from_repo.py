"""
Use both MLM and holistic scoring methods to evaluate models on BLiMP or Zorro.

"""
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

from unmasked.mlm.scoring import mlm_score_model_on_paradigm
from unmasked.holistic.scoring import holistic_score_model_on_paradigm
from unmasked import configs
from unmasked.utils import calc_accuracy_from_scores

MODEL_REPO = "phueb/BabyBERTa"  # name of huggingface model hub repository
LOWER_CASE = True  # should model be evaluated on lower-cased input?
TEST_SUITE_NAME = ['zorro', 'blimp'][0]

if TEST_SUITE_NAME == 'blimp':
    num_expected_scores = 2000
elif TEST_SUITE_NAME == 'zorro':
    num_expected_scores = 4000
else:
    raise AttributeError('Invalid "TEST_SUITE_NAME".')


# each iteration in this "for" loop produces 1 row in the data-frame
for scoring_method in ['holistic', 'mlm']:

    if scoring_method == 'mlm':
        score_model_on_paradigm = mlm_score_model_on_paradigm
    elif scoring_method == 'holistic':
        score_model_on_paradigm = holistic_score_model_on_paradigm
    else:
        raise AttributeError('Invalid scoring_method.')

    # load from repo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_REPO)

    model.eval()
    model.cuda(0)

    # for each paradigm in test suite
    accuracies = []
    for path_paradigm in (configs.Dirs.test_suites / TEST_SUITE_NAME).glob('*.txt'):

        # scoring
        print(f"Scoring {path_paradigm.name:<60} with {MODEL_REPO:<40} and method={scoring_method}")
        scores = score_model_on_paradigm(model, tokenizer, path_paradigm, lower_case=LOWER_CASE)

        assert len(scores) == num_expected_scores

        # compute accuracy
        accuracy = calc_accuracy_from_scores(scores, scoring_method)

        # collect
        accuracies.append(accuracy)

    print(f'Overall accuracy={np.mean(accuracies):.4f}')


