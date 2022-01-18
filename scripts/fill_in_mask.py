from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizerFast
import torch

MODEL_REPO = "phueb/BabyBERTa-2"  # name of huggingface model hub repository

# load from repo
tokenizer: RobertaTokenizerFast = AutoTokenizer.from_pretrained(MODEL_REPO,
                                                                add_prefix_space=True,  # this is required
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

