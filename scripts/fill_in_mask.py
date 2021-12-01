from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

MODEL_REPO = "phueb/BabyBERTa-1"  # name of huggingface model hub repository

# load from repo
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO,
                                          add_prefix_space=True,  # this must be True for BabyBERTa
                                          )
model = AutoModelForMaskedLM.from_pretrained(MODEL_REPO)

model.eval()
model.cuda(0)

SENTENCE = 'what does this <mask> say ?'


def print_predictions(sent):
    x = tokenizer(sent, return_tensors='pt')

    with torch.no_grad():
        output = model(**{k: v.to('cuda') for k, v in x.items()})
    last_hidden_state = output['logits'].squeeze()

    list_of_list = []
    for pos, input_id in enumerate(x['input_ids'].squeeze()):
        mask_hidden_state = last_hidden_state[pos]
        idx = torch.topk(mask_hidden_state, k=5, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        list_of_list.append(words)
        print(tokenizer.decode(input_id))
        print("Guesses : ", words)
        print()


print_predictions(SENTENCE)
