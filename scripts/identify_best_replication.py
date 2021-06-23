"""
Find best-performing model among replications (models trained in the same condition, differing only in initial weights)
"""

import pandas as pd
from typing import Union
from pathlib import Path

from unmasked import configs
from unmasked.utils import get_group_names

TEST_SUITE_NAME = ['zorro', 'blimp'][0]  # zorro or blimp

df_path = configs.Dirs.results / f'{TEST_SUITE_NAME}.csv'
if df_path.exists():
    df = pd.read_csv(df_path, index_col=False)
else:
    raise OSError(f'Did not find {df_path}')


pd.options.display.max_columns = None
pd.options.display.width = None


def shorten_path(p: Union[None, Path]):
    if not Path(p).exists():
        return p
    elif Path(p).name.startswith('checkpoint'):  # fairseq model
        return Path(p).relative_to(configs.Dirs.fairseq_models)
    else:
        return Path(p).relative_to(configs.Dirs.babyberta_runs)


df['path'] = df['path'].apply(shorten_path)


# print summary statistics
summary = df.groupby(['model', 'corpora', 'scoring_method']).agg(
    {'overall': ['min', 'max', 'mean', 'count']}).sort_values(axis=0, by=('overall', 'mean')).round(1)
print(summary)

# group names
group_names = get_group_names(df)  # this creates "group_name" column in df

# print accuracy of each rep, in each group
for group_name, grouped in df.groupby(['group_name']):
    print()
    print(group_name)
    if group_name.startswith('BabyBERTa'):
        index_name = 'path'  # only BabyBERTa has path
    else:
        index_name = 'rep'
    cols = [index_name, 'scoring_method', 'overall']
    group_summary = grouped[cols].pivot(index=index_name, columns='scoring_method', values='overall')
    print(group_summary.sort_values(axis=0, by='holistic'))

