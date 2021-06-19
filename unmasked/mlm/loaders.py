""""
Adapted from https://github.com/awslabs/mlm-scoring
"""
from collections import OrderedDict

from typing import Any, Iterable, List, TextIO, Tuple


class Corpus(OrderedDict):
    """A ground truth corpus (dictionary of ref sentences)
    """

    @classmethod
    def from_file(cls, fp: TextIO, **kwargs) -> Any:
        # A text file (probably LM training data per line)
        return Corpus.from_text(fp, **kwargs)

    @classmethod
    def from_text(cls, fp: Iterable[str], lower_case=False):
        corpus = cls()
        # For text files, utterance ID is just the zero-indexed line number
        idx = 0
        for line in fp:
            if lower_case:
                corpus[idx] = line.strip().lower()
            else:
                corpus[idx] = line.strip()

            idx += 1
        return corpus

    def _get_num_words_in_utt(self, utt: str) -> int:
        return len(utt.split(' '))

    def get_num_words(self) -> Tuple[List[int], int]:

        max_words_per_utt = 0
        num_words_list = []

        for utt_id, utt in self.items():
            num_words = self._get_num_words_in_utt(utt)
            num_words_list.append(num_words)
            if num_words > max_words_per_utt:
                max_words_per_utt = num_words

        return num_words_list, max_words_per_utt
