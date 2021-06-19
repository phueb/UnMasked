"""
Use both MLM and holistic scoring methods to evaluate models on BLiMP or Zorro.

WITHOUT capitalization of proper nouns, average accuracies on zorro:
RoBERTa-baseAO-CHILDES           73.64
BabyBERTa+Wikipedia-1            73.82
BabyBERTa+AO-CHILDES+50Kvocab    77.11
BabyBERTa+AO-Newsela             77.31
RoBERTa-base+10M                 78.47
BabyBERTa+AO-CHILDES             78.93
RoBERTa-base+Wikipedia-1         79.78
BabyBERTa+concatenated           86.5
RoBERTa-base+30B                 90.63

WITH capitalization of proper nouns, average accuracies on zorro:
RoBERTa-baseAO-CHILDES           72.24 (down because proper nouns are lower-cased in training data  and tokenizer does not lower case during eval)
BabyBERTa+Wikipedia-1            73.82 (same)
BabyBERTa+AO-CHILDES+50Kvocab    75.82 (down)
BabyBERTa+AO-Newsela             77.31 (same)
RoBERTa-base+Wikipedia-1         78.25 (down because proper nouns are lower-cased in training data and tokenizer does not lower case during eval)
BabyBERTa+AO-CHILDES             78.93 (same)
RoBERTa-base+10M                 79.55 (up)
BabyBERTa+concatenated           86.5  (same)
RoBERTa-base+30B                 91.13 (up)
"""
import shutil

from unmasked.mlm.scoring import mlm_score_model_on_paradigm
from unmasked.holistic.scoring import holistic_score_model_on_paradigm
from unmasked import configs
from unmasked.models import load_babyberta_models, load_roberta_base_models

BABYBERTA_PARAMS = [1]  # which BabyBerta models to load
OVERWRITE = False  # set to True to remove existing scores and re-score
TEST_SUITE_NAME = 'zorro'  # zorro or blimp
LOWER_CASE = True  # this affects both zorro (proper nouns) and blimp

if TEST_SUITE_NAME == 'blimp':
    num_paradigms = 67
elif TEST_SUITE_NAME == 'zorro':
    num_paradigms = 23
else:
    raise AttributeError('Invalid "TEST_SUITE_NAME".')


# load all models
name2model = {}
name2model.update(load_babyberta_models(BABYBERTA_PARAMS))
# name2model.update(load_roberta_base_models())

for model_name, (model, tokenizer) in name2model.items():

    print(f'model_name={model_name}')

    path_model = configs.Dirs.results / TEST_SUITE_NAME / f'lower_case={LOWER_CASE}' / model_name

    model.eval()
    model.cuda(0)

    # compute and save scores
    for path_paradigm in (configs.Dirs.test_suites / TEST_SUITE_NAME).glob('*.txt'):

        for scoring_method in ['mlm', 'holistic']:

            if scoring_method == 'mlm':
                score_model_on_paradigm = mlm_score_model_on_paradigm
            elif scoring_method == 'holistic':
                score_model_on_paradigm = holistic_score_model_on_paradigm
            else:
                raise AttributeError('Invalid scoring_method.')

            path_results_file = path_model / scoring_method / path_paradigm.name

            if path_results_file.exists() and not OVERWRITE:
                continue

            if not path_results_file.parent.exists():
                path_results_file.parent.mkdir(parents=True)

            # scoring
            print(f"Scoring pairs in {path_paradigm.name:<60} with {model_name:<60} and method={scoring_method}")
            scores = score_model_on_paradigm(model, tokenizer, path_paradigm, lower_case=LOWER_CASE)
            with path_results_file.open('w') as f:
                f.write('\n'.join([str(score) for score in scores]))

