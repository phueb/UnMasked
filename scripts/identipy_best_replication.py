"""
Find best-performing model among replications (models trained in the same condition, differing only in initial weights)
"""

import pandas as pd

from unmasked import configs
from unmasked.utils import get_group_names

TEST_SUITE_NAME = ['zorro', 'blimp'][0]  # zorro or blimp

df_path = configs.Dirs.results / f'{TEST_SUITE_NAME}.csv'
if df_path.exists():
    df = pd.read_csv(df_path, index_col=False)
else:
    raise OSError(f'Did not find {df_path}')


# print summary statistics
summary = df.groupby(['model', 'corpora', 'scoring_method']).agg(
    {'overall': ['min', 'max', 'median']}).sort_values(axis=0, by=('overall', 'median')).round(1)
print(summary)

# group names
group_names = get_group_names(df)  # this creates "group_name" column in df

# print accuracy of each rep, in each group
for group_name, grouped in df.groupby(['group_name']):
    print()
    print(group_name)
    cols = ['rep', 'scoring_method', 'overall']
    group_summary = grouped[cols].pivot(index='rep', columns='scoring_method', values='overall')
    print(group_summary.sort_values(axis=0, by='holistic'))

    # TODO identify best param_name for BabyBERTa
