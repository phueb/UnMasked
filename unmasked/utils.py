from typing import List


def calc_accuracy_from_scores(scores: List[float],
                              scoring_method: str,
                              ) -> float:
    """
    Compute accuracy given scores.

    Notes:
        Odd-numbered lines contain grammatical sentences, an even-numbered lines contain un-grammatical sentences.

        Whether a pair is scored as correct depends on the scoring method:
        When scoring with MLM method, the pseudo-log-likelihood must be larger for the correct sentence.
        When scoring with the holistic method, the cross-entropy error sum must be smaller for the correct sentence
    """

    num_pairs = len(scores) // 2
    num_correct = 0
    for i in range(num_pairs):
        if scoring_method == 'mlm' and float(scores[2 * i]) > float(scores[2 * i + 1]):
            num_correct += 1
        elif scoring_method == 'holistic' and float(scores[2 * i]) < float(scores[2 * i + 1]):
            num_correct += 1

    return num_correct / num_pairs * 100
