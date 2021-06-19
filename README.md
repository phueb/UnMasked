# UnMasked

Score masked language models on grammatical knowledge test suites.

## Test Suites

- BLiMP
- Zorro

## Models

- BabyBERTa
- RoBERTa-base

## Scoring Methods

- holistic scoring (i.e. sum of cross-entropy error at every token)
- MLM-scoring (i.e. sum of pseudo-log-likelihoods)

MLM-scoring was proposed by Salazar et al., 2013. 
This method computes pseudo-log-likelihoods and requires masking each word in the input one-at-a-time. 
In contrast, the holistic scoring procedure does not use mask symbols, 
and instead computes the sum of the cross-entropy errors for every token in the input.

These two methods produce very different results. 
Our methods favors models trained without predicting unmasked tokens, and handicaps those that were trained in this way (all Roberta models save for BabyBERTa).
The pseudo-log-likelihood method does not handicap models trained with predicting unmasked tokens, because it uses mask symbols to compute scores, 
which ensures that a model never has access to information in the input about what word it should predict.

