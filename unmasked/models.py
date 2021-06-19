"""
Load all models. 
There are three different models:
1. BabyBERTa trained with huggingface transformers by us. (loaded from a restricted remote drive)
2. RoBERTa-base trained in fairseq by us.
3. RoBERTa-base trained by others and loaded using huggingface transformers.
"""

import yaml
from typing import List, Tuple, Dict

from fairseq.models.roberta import RobertaModel
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM

from unmasked import configs
from unmasked.convert_fairseq_roberta import convert_fairseq_roberta_to_pytorch


def load_babyberta_models(param_numbers: List[int],
                          ) -> Dict[str, Tuple[RobertaForMaskedLM, RobertaTokenizerFast]]:
    """
    load all BabyBERTa models trained by us.

    notes:
    - models are stored on shared remote drive with restricted access.
    - there are multiple replications of each model (with different initializations)
    """
    res = {}
    for param_name in [f'param_{i:03}' for i in param_numbers]:
        for path_to_babyberta in (configs.Dirs.babyberta_runs / param_name).glob('**/saves/'):
            with (configs.Dirs.babyberta_runs / param_name / 'param2val.yaml').open('r') as f:
                param2val = yaml.load(f, Loader=yaml.FullLoader)
            if param2val['corpora'] == ('aochildes',):
                corpora = 'AO-CHILDES'
            elif param2val['corpora'] == ('wikipedia-1',):
                corpora = 'Wikipedia-1'
            elif param2val['corpora'] == ('aonewsela',):
                corpora = 'AO-Newsela'
            else:
                raise NotImplementedError
            rep_name = path_to_babyberta.parent.name
            model_name = f'BabyBERTa_{corpora}+{rep_name}'
            model_path = configs.Dirs.babyberta_runs / param_name / rep_name / 'saves'
            model = RobertaForMaskedLM.from_pretrained(str(model_path))
            tokenizer = RobertaTokenizerFast.from_pretrained(str(model_path),
                                                             add_prefix_space=True,  # this must be added
                                                             )
            res[model_name] = (model, tokenizer)

    return res


def load_fairseq_as_huggingface_models() -> Dict[str, Tuple[RobertaForMaskedLM, RobertaTokenizerFast]]:
    """load Roberta-base models pre-trained by us, and convert to huggingface Roberta """

    res = {}
    for path_model_data in configs.Dirs.fairseq_models.glob('*'):
        for checkpoint_fn in path_model_data.glob(f'checkpoint_last+rep=*.pt'):

            if 'AO-CHILDES' in path_model_data.name:
                bin_name = 'aochildes-data-bin'
                corpora = 'AO-CHILDES'
            elif 'Wikipedia-1' in path_model_data.name:
                bin_name = 'wikipedia1_new1_seg'
                corpora = 'Wikipedia-1'
            else:
                raise AttributeError('Invalid model name.')

            model_ = RobertaModel.from_pretrained(model_name_or_path=str(path_model_data),
                                                  checkpoint_file=str(path_model_data / checkpoint_fn),
                                                  data_name_or_path=str(path_model_data / bin_name),
                                                  )
            # collect
            rep_name = checkpoint_fn.stem.split('+')[-1]
            model_name = f'RoBERTa-base_{corpora}+{rep_name}'
            res[model_name] = convert_fairseq_roberta_to_pytorch(model_)
    return res


def load_roberta_base_models():

    # pre-trained by others
    huggingface_models = {
        'RoBERTa-base_Warstadt2020+rep=0': (AutoModelForMaskedLM.from_pretrained("nyu-mll/roberta-base-10M-1"),
                                            AutoTokenizer.from_pretrained("nyu-mll/roberta-base-10M-2")),
        'RoBERTa-base_Warstadt2020+rep=1': (AutoModelForMaskedLM.from_pretrained("nyu-mll/roberta-base-10M-2"),
                                            AutoTokenizer.from_pretrained("nyu-mll/roberta-base-10M-2")),
        'RoBERTa-base_Warstadt2020+rep=2': (AutoModelForMaskedLM.from_pretrained("nyu-mll/roberta-base-10M-3"),
                                            AutoTokenizer.from_pretrained("nyu-mll/roberta-base-10M-2")),
        'RoBERTa-base_Liu2019+rep=0': (RobertaForMaskedLM.from_pretrained('roberta-base'),
                                       RobertaTokenizerFast.from_pretrained('roberta-base')),
    }

    # trained by us using fairseq - need to be converted
    fairseq_models = load_fairseq_as_huggingface_models()

    res = {}
    res.update(huggingface_models)
    res.update(fairseq_models)
    return res
