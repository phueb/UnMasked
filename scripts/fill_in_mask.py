
"""
When using BabyBERTa to fill-in masked tokens, the automated pipeline by huggingface produces incorrect results.
The tokenizer that babyberta uses treats the space before the <mask> as a token, and the <mask> as another token.
This probably has something to do with add_prefix_space=True.
(I set add_prefix_space=True because babyberta is supposed to be a model of spoken language which does not distinguish
 between title-cased and non-title-cased words.)
At any rate, the predictions for the masked word are output by babyberta at the space token,
 the token that is before the mask symbol. The predictions at this position look good.
For example, below are the top 5 outputs for each token in the sentence "what does this <mask> do?".
The tokens the tokenizer generated are shown in the first column, from top to bottom.
The predictions are shown in the Python list next to each input token.
This explains the strange predictions for <mask>:
The model is trying to complete its previous predictions with a sub-word,
 hence the absence of Ġ, which simply means white-space.

<s>              ['Ġ?              ', 'Ġ.              ', 'Ġ!              ', 'Ġfirst          ', 'Ġc              ']
Ġwhat            ['Ġwhat           ', 'Ġhow            ', 'Ġoh             ', 'Ġwhy            ', 'Ġokay           ']
Ġdoes            ['Ġdoes           ', "'s              ", 'Ġdid            ', 'Ġis             ', 'Ġshould         ']
Ġthis            ['Ġa              ', 'Ġthe            ', 'Ġthis           ', 'Ġthat           ', 'Ġyour           ']
Ġ                ['Ġlittle         ', 'Ġone            ', 'Ġj              ', 'Ġpig            ', 'Ġguy            '] <-
<mask>           ['ick             ', 'ore             ', 'ig              ', 'ee              ', 'ill             ']
Ġdo              ['Ġdo             ', 'Ġlike           ', 'Ġhave           ', 'Ġsay            ', 'Ġfor            ']
Ġ?               ['Ġ?              ', 'Ġ.              ', 'Ġ!              ', 'Ġc              ', 'Ġfirst          ']
</s>             ['Ġ?              ', 'Ġ.              ', 'Ġ!              ', 'Ġfirst          ', 'Ġc              ']
"""
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

MODEL_REPO = "phueb/BabyBERTa-2"  # name of huggingface model hub repository

# load from repo
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO,
                                          add_prefix_space=True,  # this must be True for BabyBERTa
                                          )
model = AutoModelForMaskedLM.from_pretrained(MODEL_REPO)

model.eval()
model.cuda(0)

SENTENCE = 'what does this <mask> do ?'


def print_predictions(sent):
    x = tokenizer(sent, return_tensors='pt')

    with torch.no_grad():
        output = model(**{k: v.to('cuda') for k, v in x.items()})
    logits = output['logits'].squeeze()

    for pos, input_id in enumerate(x['input_ids'].squeeze()):
        logits_for_token = logits[pos]
        idx = torch.topk(logits_for_token, k=5, dim=0)[1]
        predictions = [f'{tokenizer.decode(i.item()).strip():<16}' for i in idx]
        input_token = tokenizer.decode(input_id)
        print(f'{input_token:<16} {predictions}')


print_predictions(SENTENCE)

