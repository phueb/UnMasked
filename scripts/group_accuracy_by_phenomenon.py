import pandas as pd

from unmasked import configs
from unmasked.helpers import blimp_phenomenon2fns, blimp_phenomena
from unmasked.helpers import zorro_phenomenon2fns, zorro_phenomena
from unmasked.helpers import shorten_phenomenon

TEST_SUITE_NAME = 'zorro'  # zorro or blimp

if TEST_SUITE_NAME == 'blimp':
    phenomenon2fns = blimp_phenomenon2fns
    phenomena = blimp_phenomena
elif TEST_SUITE_NAME == 'zorro':
    phenomenon2fns = zorro_phenomenon2fns
    phenomena = zorro_phenomena
else:
    raise AttributeError('Invalid TEST_SUITE_NAME.')

pd.options.display.max_columns = None
pd.options.display.width = None

df_path = configs.Dirs.results / f'{TEST_SUITE_NAME}.csv'
if df_path.exists():
    df = pd.read_csv(df_path, index_col=False)
else:
    raise OSError(f'Did not find {df_path}')

# group accuracies by phenomena
df_phenomena = df[['model', 'corpora', 'rep', 'scoring_method']].copy()
for phenomenon, fns in phenomenon2fns.items():

    col = df[fns].mean(axis=1).round(1)
    df_phenomena[phenomenon] = col


# print overall accuracy
summary = df_phenomena.groupby(['model', 'corpora', 'scoring_method'])[phenomena].mean()
summary['overall'] = summary.mean(axis=1)
summary = summary.sort_values(axis=0, by='overall')['overall'].round(1)
print()
print(summary)
