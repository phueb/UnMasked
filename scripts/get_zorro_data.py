"""
Get minimal pairs from Zorro and move them here,
such that each odd-numbered line contains a grammatical sentence,
and each even-numbered line contains an un-grammatical sentence, following BLiMP.
"""
from pathlib import Path
from typing import List

from unmasked import configs


PATH_TO_ZORRO_SENTENCES = Path().home() / 'Zorro' / 'sentences' / 'babyberta'


def reverse_even_odd_lines(lines: List[str]):
    for i in range(0, len(lines) - 1, 2):
        yield lines[i + 1]
        yield lines[i + 0]


for path_to_paradigm in PATH_TO_ZORRO_SENTENCES.glob('*.txt'):
    print(f'Moving sentences from {path_to_paradigm.name}')

    sentences = path_to_paradigm.read_text().split('\n')[:-1]

    sentences_reordered = [s for s in reverse_even_odd_lines(sentences)]

    assert len(sentences_reordered) == len(sentences)
    assert len(set(sentences_reordered)) == len(set(sentences))

    # copy
    dest = configs.Dirs.test_suites / 'zorro' / path_to_paradigm.name
    text = '\n'.join(sentences_reordered)
    dest.write_text(text)