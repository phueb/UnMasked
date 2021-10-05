<div align="center">
 <img src="images/logo.png" width="250"> 
 <br>
 Score masked language models on grammatical knowledge test suites.
</div>


## Test Suites

- [BLiMP](https://github.com/alexwarstadt/blimp)
- [Zorro](https://github.com/phueb/Zorro)

## Models

- BabyBERTa
- RoBERTa-base

## Scoring Methods

- holistic scoring (i.e. sum of cross-entropy error at every token)
- MLM-scoring (i.e. sum of pseudo-log-likelihoods)

MLM-scoring was proposed by [Salazar et al., 2019](https://arxiv.org/abs/1910.14659). 
This method computes pseudo-log-likelihoods and requires masking each word in the input one-at-a-time. 
In contrast, the holistic scoring proposed by [Zaczynska et al., 2020](https://arxiv.org/abs/2007.03765) procedure does not use mask symbols, 
and instead computes the sum of the cross-entropy errors for every token in the input.

These two methods produce very different results. 
The holistic method favors models trained without predicting unmasked tokens, 
and handicaps those that were trained in this way (all Roberta models save for BabyBERTa).
The 'MLM" method does not handicap models trained with predicting unmasked tokens, 
because it uses mask symbols to compute scores, 
which ensures that a model never has access to information in the input about what word it should predict.

