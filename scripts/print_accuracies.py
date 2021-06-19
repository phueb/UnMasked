""""
Collect scores saved in results folder, and print accuracies.

Odd-numbered lines contain grammatical sentences, an even-numbered lines contain un-grammatical sentences.

Whether a pair is scored as correct depends on the scoring method:
When scoring with MLM method, the pseudo-log-likelihood must be larger for the correct sentence.
When scoring with the holistic method, the cross-entropy error sum must be smaller for the correct sentence
"""
import numpy as np
import pandas as pd
from collections import defaultdict

from unmasked import configs


test_suite_name = 'zorro'
lower_case = True

if test_suite_name == 'blimp':
    from unmasked.helpers import blimp_file_name2phenomenon as file_name2phenomenon
    num_expected_pairs = 1000
elif test_suite_name == 'zorro':
    from unmasked.helpers import zorro_file_name2phenomenon as file_name2phenomenon
    num_expected_pairs = 2000
else:
    raise AttributeError('Invalid arg to "data_name".')
phenomena = set([v for v in file_name2phenomenon.values()])
phenomenon2fns = {phenomenon: [pa for pa, ph in file_name2phenomenon.items() if ph == phenomenon]
                  for phenomenon in phenomena}

path_to_results_dir = configs.Dirs.results / test_suite_name / f'lower_case={lower_case}'
phenomenon2col = defaultdict(list)
for model_dir in path_to_results_dir.glob('*'):

    model_name, rep_name = model_dir.name.split('+')

    for scoring_method in ['mlm', 'holistic']:

        # each model replication has two rows, one for each scoring method
        phenomenon2col['Model'].append(model_name.replace('_', '+'))
        phenomenon2col['rep'].append(rep_name)
        phenomenon2col['scoring_method'].append(scoring_method)

        for phenomenon, fns in phenomenon2fns.items():

            accuracies = []
            for fn in fns:
                file = model_dir / scoring_method / f'{fn}.txt'
                with file.open('rt') as f:
                    scores = f.readlines()
                num_pairs = len(scores) // 2

                # compute accuracy for paradigm
                num_correct = 0
                for i in range(num_pairs):
                    if scoring_method == 'mlm' and float(scores[2*i]) > float(scores[2*i+1]):
                        num_correct += 1
                    elif scoring_method == 'holistic' and float(scores[2*i]) < float(scores[2*i+1]):
                        num_correct += 1

                assert num_pairs == num_expected_pairs
                acc = num_correct / num_pairs
                accuracies.append(acc * 100)

            # collect
            phenomenon2col[phenomenon.capitalize()].append(np.mean(accuracies))
            # per-class and overall accuracies are the desired (micro)averages

df = pd.DataFrame(data=phenomenon2col)
# sort columns
df = df[['Model'] + [col_name for col_name in sorted(df.columns) if col_name != 'Model']]

for scoring_method in ['mlm', 'holistic']:
    print()
    print(f'scoring_method={scoring_method}')
    print(f'test_suite={test_suite_name}')
    print(f'lower_case={lower_case}')
    df_ = df[df['scoring_method'] == scoring_method].copy()
    df_['Overall'] = df_.mean(axis=1)
    df_ = df_.sort_values(axis=0, by='Overall')
    for model_name, scoring_method, overall_acc in zip(df_['Model'], df_['scoring_method'], df_['Overall'].round(2)):
        print(f'{model_name:<40} {scoring_method:<12} {overall_acc}')

    df_.to_csv(configs.Dirs.results / f'{test_suite_name}-{lower_case}.csv')