import pandas as pd

from unmasked import configs


TEST_SUITE_NAME = 'zorro'  # zorro or blimp

if TEST_SUITE_NAME == 'blimp':
    from unmasked.helpers import blimp_file_name2phenomenon as file_name2phenomenon
    num_expected_pairs = 1000
    num_expected_paradigms = 67
elif TEST_SUITE_NAME == 'zorro':
    from unmasked.helpers import zorro_file_name2phenomenon as file_name2phenomenon
    num_expected_pairs = 2000
    num_expected_paradigms = 23


df_path = configs.Dirs.results / f'{TEST_SUITE_NAME}.csv'
if df_path.exists():
    df = pd.read_csv(df_path)
else:
    raise OSError(f'Did not find {df_path}')

print(df)

# todo group paradigms into phenomena
phenomenon = file_name2phenomenon[path_paradigm.stem]