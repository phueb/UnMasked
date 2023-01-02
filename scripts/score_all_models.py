"""
Use both MLM and holistic scoring methods to evaluate models on BLiMP or Zorro.

WITHOUT capitalization of proper nouns, average accuracies on zorro with MLM scoring:
RoBERTa-base+AO-CHILDES          73.64
BabyBERTa+Wikipedia-1            73.82
BabyBERTa+AO-CHILDES+50Kvocab    77.11
BabyBERTa+AO-Newsela             77.31
RoBERTa-base+10M                 78.47
BabyBERTa+AO-CHILDES             78.93
RoBERTa-base+Wikipedia-1         79.78
BabyBERTa+concatenated           86.5
RoBERTa-base+30B                 90.63

WITH capitalization of proper nouns, average accuracies on zorro with MLM scoring:
RoBERTa-base+AO-CHILDES          72.24 (down because proper nouns are lower-cased in training data  and tokenizer does not lower case during eval)
BabyBERTa+Wikipedia-1            73.82 (same)
BabyBERTa+AO-CHILDES+50Kvocab    75.82 (down)
BabyBERTa+AO-Newsela             77.31 (same)
RoBERTa-base+Wikipedia-1         78.25 (down because proper nouns are lower-cased in training data and tokenizer does not lower case during eval)
BabyBERTa+AO-CHILDES             78.93 (same)
RoBERTa-base+10M                 79.55 (up)
BabyBERTa+concatenated           86.5  (same)
RoBERTa-base+30B                 91.13 (up)
"""
import pandas as pd
from collections import defaultdict
import torch

from unmasked.mlm.scoring import mlm_score_model_on_paradigm
from unmasked.holistic.scoring import holistic_score_model_on_paradigm
from unmasked import configs
from unmasked.utils import calc_accuracy_from_scores
from unmasked.models import load_babyberta_models, load_roberta_base_models, ModelData

BABYBERTA_PARAMS = [16, 17] + [12, 13] + [14, 15] + [1, 2, 3, 7]  # which BabyBerta models to load from shared drive
OVERWRITE = False  # set to True to remove existing scores and re-score
TEST_SUITE_NAME = ['zorro', 'blimp'][0]

if TEST_SUITE_NAME == 'blimp':
    num_expected_scores = 2000
elif TEST_SUITE_NAME == 'zorro':
    num_expected_scores = 4000
else:
    raise AttributeError('Invalid "TEST_SUITE_NAME".')

# load previously generated dataframe with model accuracies
df_path = configs.Dirs.results / f'{TEST_SUITE_NAME}.csv'
if df_path.exists() and not OVERWRITE:
    df_old = pd.read_csv(df_path, index_col=False)
else:
    df_old = pd.DataFrame()

# load all models
models_data = []
models_data.extend(load_roberta_base_models())
models_data.extend(load_babyberta_models(BABYBERTA_PARAMS))

sub_dfs = [df_old]
for model_data in models_data:

    model_data: ModelData

    print()

    # each iteration in this "for" loop produces 1 row in the data-frame
    for scoring_method in ['holistic', 'mlm']:

        if scoring_method == 'mlm':
            score_model_on_paradigm = mlm_score_model_on_paradigm
        elif scoring_method == 'holistic':
            score_model_on_paradigm = holistic_score_model_on_paradigm
        else:
            raise AttributeError('Invalid scoring_method.')

        # skip scoring if row exists in data frame
        # if not OVERWRITE and df_old.any(axis=None):
        #     bool_id = (df_old['model'].str.contains(model_data.name)) & \
        #               (df_old['corpora'].str.contains(model_data.corpora)) & \
        #               (df_old['rep'] == model_data.rep) & \
        #               (df_old['path'].str.contains(model_data.path)) & \
        #               (df_old['scoring_method'].str.contains(scoring_method))
        #     if df_old[bool_id].any(axis=None):
        #         print(f'Skipping {model_data.name:<40} corpora={model_data.corpora:<40} rep={model_data.rep:<6}')
        #         continue

        # TODO duplicate rows will happen when this script is run once and then again after more BabyBERTa replications
        #  are added to a folder which was previously searched, because the variable "rep" will be assigned differently
        #  to an already evaluated model

        print(f'Scoring  {model_data.name:<40} corpora={model_data.corpora:<40} rep={model_data.rep:<6}')

        # collect model scoring data
        model_accuracy_data = defaultdict(list)
        model_accuracy_data['model'].append(model_data.name)
        model_accuracy_data['corpora'].append(model_data.corpora)
        model_accuracy_data['rep'].append(model_data.rep)
        model_accuracy_data['path'].append(model_data.path)
        model_accuracy_data['scoring_method'].append(scoring_method)

        # should model be evaluated on lower-cased input?
        if model_data.case_sensitive and model_data.trained_on_lower_cased_data:
            lower_case = True
            print('Will score with lower-cased input.')
        else:
            print('Will not score with lower-cased input.')
            lower_case = False

        # call the function that loads the model and tokenizer
        model = model_data.model()
        tokenizer = model_data.tokenizer()

        model.eval()
        model.cuda(0)

        # for each paradigm in test suite
        col_names_with_acc = []
        for path_paradigm in (configs.Dirs.test_suites / TEST_SUITE_NAME).glob('*.txt'):

            # scoring
            print(f"Scoring {path_paradigm.name:<60} with {model_data.name:<40} trained on {model_data.corpora:<40} and method={scoring_method}")
            scores = score_model_on_paradigm(model, tokenizer, path_paradigm, lower_case=lower_case)

            assert len(scores) == num_expected_scores

            # compute accuracy
            accuracy = calc_accuracy_from_scores(scores, scoring_method)

            # collect
            model_accuracy_data[path_paradigm.stem].append(accuracy)
            col_names_with_acc.append(path_paradigm.stem)

        # clear gpu memory
        del model
        torch.cuda.empty_cache()

        # prepare and collect sub-dataframe
        df_sub = pd.DataFrame(data=model_accuracy_data)
        df_sub = df_sub[['model', 'corpora', 'rep', 'path', 'scoring_method'] + col_names_with_acc]  # sort columns
        df_sub['overall'] = df_sub[col_names_with_acc].mean(axis=1)
        sub_dfs.append(df_sub)

        print(df_sub[['model', 'corpora', 'rep', 'scoring_method', 'overall']])

        # save combined data frame
        df = pd.concat(sub_dfs, axis=0, sort=True)
        df = df.drop_duplicates(subset=['model', 'corpora', 'rep', 'path', 'scoring_method'])
        df = df.sort_values(axis=0, by='overall')
        df = df.round(2)
        df.to_csv(configs.Dirs.results / f'{TEST_SUITE_NAME}.csv', index=False)
