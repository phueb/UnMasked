"""
compare accuracy between models saved in local or remote runs folder, at one time step.
"""

import pandas as pd
import numpy as np
from scipy.stats import sem, t

from unmasked.utils import get_group_names
from unmasked.visualizer import Visualizer, ParadigmData
from unmasked.helpers import zorro_file_name2phenomenon
from unmasked.helpers import blimp_file_name2phenomenon
from unmasked import configs

TEST_SUITE_NAME: str = ['zorro', 'blimp'][0]
SCORING_METHOD: str = ['holistic', 'mlm'][0]

# load accuracy data
df_path = configs.Dirs.results / f'{TEST_SUITE_NAME}.csv'
if df_path.exists():
    df = pd.read_csv(df_path, index_col=False)
else:
    raise OSError(f'Did not find {df_path}')

if TEST_SUITE_NAME == 'zorro':
    num_paradigms = 23
    file_name2phenomenon = zorro_file_name2phenomenon
elif TEST_SUITE_NAME == 'blimp':
    num_paradigms = 67
    file_name2phenomenon = blimp_file_name2phenomenon
else:
    raise AttributeError('Invalid TEST_SUITE_NAME.')


# remove rows with scoring_method other than which is requested
df = df[df['scoring_method'] == SCORING_METHOD]

# group names
group_names = get_group_names(df)

# collects and plots each ParadigmData instance in 1 multi-axis figure
title = f'test suite={TEST_SUITE_NAME}\nscoring method={SCORING_METHOD}'
v = Visualizer(num_paradigms, group_names, title)

# for all paradigms
for fn, phenomenon in file_name2phenomenon.items():

    # for each group, get accuracies for each replication
    group_name2accuracies = {gn: df[df['group_name'] == gn][fn].values for gn in group_names}
    print(group_name2accuracies)

    paradigm = fn.replace('_', ' ').lstrip(phenomenon).lstrip('-')

    pd = ParadigmData(
        phenomenon=phenomenon,
        paradigm=paradigm,
        group_name2accuracies=group_name2accuracies
    )

    # plot each paradigm in separate axis
    v.update(pd)

v.plot_summary()

# print summary
print()
print(f'Average accuracies on test_suite={TEST_SUITE_NAME} and with scoring_method={SCORING_METHOD}')
for group_name in group_names:
    acc_mat = np.array([pd.group_name2accuracies[group_name] for pd in v.pds])  # (num paradigms, num reps)
    accuracies_by_rep = acc_mat.mean(axis=0)  # (num reps)
    # margin of error (across replications)
    n = len(accuracies_by_rep)
    h = sem(accuracies_by_rep, axis=0) * t.ppf((1 + v.confidence) / 2, n - 1) if n > 1 else np.nan
    print(f'{group_name:.<64}accuracy={accuracies_by_rep.mean().item():.2f} n={n:>2} margin-of-error={h:>8.2f}')

