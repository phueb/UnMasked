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
import pandas as pd
from collections import defaultdict

from unmasked.mlm.scoring import mlm_score_model_on_paradigm
from unmasked.holistic.scoring import holistic_score_model_on_paradigm
from unmasked import configs
from unmasked.utils import calc_accuracy_from_scores
from unmasked.models import load_babyberta_models, load_roberta_base_models, ModelData

BABYBERTA_PARAMS = [1, 2, 3, 7]  # which BabyBerta models to load from shared drive with restricted access
OVERWRITE = True  # set to True to remove existing scores and re-score
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
models_data.extend(load_babyberta_models(BABYBERTA_PARAMS))
models_data.extend(load_roberta_base_models())

sub_dfs = [df_old]
for model_data in models_data:

    model_data: ModelData
    model = model_data.model
    tokenizer = model_data.tokenizer

    # skip scoring if accuracies already exists in data frame
    if not OVERWRITE and df_old.any(axis=None):
        bool_id = (df_old['model'].str.contains(model_data.name)) & \
              (df_old['corpora'].str.contains(model_data.corpora)) & \
              (df_old['rep'] == model_data.rep)
        if df_old[bool_id].any(axis=None):
            print(f'Skipping {model_data.name:<40} corpora={model_data.corpora:<40} rep={model_data.rep:<6}')
            continue

    print()

    model.eval()
    model.cuda(0)

    # init data for data-frame
    model_accuracy_data = defaultdict(list)

    # for each scoring method
    for scoring_method in ['mlm', 'holistic']:

        if scoring_method == 'mlm':
            score_model_on_paradigm = mlm_score_model_on_paradigm
        elif scoring_method == 'holistic':
            score_model_on_paradigm = holistic_score_model_on_paradigm
        else:
            raise AttributeError('Invalid scoring_method.')

        # collect model scoring data
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

        # for each paradigm in test suite
        for path_paradigm in (configs.Dirs.test_suites / TEST_SUITE_NAME).glob('*.txt'):

            # scoring
            print(f"Scoring {path_paradigm.name:<60} with {model_data.name + model_data.corpora:<40} and method={scoring_method}")
            scores = score_model_on_paradigm(model, tokenizer, path_paradigm, lower_case=lower_case)

            assert len(scores) == num_expected_scores

            # compute accuracy
            accuracy = calc_accuracy_from_scores(scores, scoring_method)

            # collect accuracy
            model_accuracy_data[path_paradigm.stem].append(accuracy)

    # prepare and collect sub-dataframe
    df_sub = pd.DataFrame(data=model_accuracy_data)
    df_sub = df_sub[['model'] +
                    [col_name for col_name in sorted(df_sub.columns) if col_name != 'model']]  # sort columns
    df_sub['overall'] = df_sub.mean(axis=1)
    sub_dfs.append(df_sub)

    print(df_sub[['model', 'corpora', 'rep', 'scoring_method', 'overall']])

    # save combined data frame
    df = pd.concat(sub_dfs, axis=0)
    df = df.drop_duplicates()
    df = df.sort_values(axis=0, by='overall')
    df = df.round(2)
    df.to_csv(configs.Dirs.results / f'{TEST_SUITE_NAME}.csv', index=False)
