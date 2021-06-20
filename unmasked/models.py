"""
Load all models. 
There are three different models:
1. BabyBERTa trained with huggingface transformers by us. (loaded from a restricted remote drive)
2. RoBERTa-base trained in fairseq by us.
3. RoBERTa-base trained by others and loaded using huggingface transformers.
"""

import yaml
from typing import List
from dataclasses import dataclass

from fairseq.models.roberta import RobertaModel
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM

from unmasked import configs
from unmasked.convert_fairseq_roberta import convert_fairseq_roberta_to_pytorch


@dataclass
class ModelData:
    name: str
    corpora: str
    rep: str
    model: RobertaForMaskedLM
    tokenizer: RobertaTokenizerFast

    # use these two attributes to decide if to evaluate model on lower-cased test sentences
    case_sensitive: bool
    trained_on_lower_cased_data: bool


def load_babyberta_models(param_numbers: List[int],
                          ) -> List[ModelData]:
    """
    load all BabyBERTa models trained by us.

    notes:
    - models are stored on shared remote drive with restricted access.
    - there are multiple replications of each model (with different initializations)
    """
    res = []
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
            rep = path_to_babyberta.parent.name
            model_path = configs.Dirs.babyberta_runs / param_name / rep / 'saves'
            model = RobertaForMaskedLM.from_pretrained(str(model_path))
            tokenizer = RobertaTokenizerFast.from_pretrained(str(model_path),
                                                             add_prefix_space=True,  # this must be added
                                                             )

            model_data = ModelData(name='BabyBERTa',
                                   corpora=corpora,
                                   rep=rep,
                                   model=model,
                                   tokenizer=tokenizer,
                                   case_sensitive=False,
                                   trained_on_lower_cased_data=True,
                                   )
            res.append(model_data)

    return res


def load_fairseq_as_huggingface_models() -> List[ModelData]:
    """load Roberta-base models pre-trained by us, and convert to huggingface Roberta """

    res = []
    for path_model_data in configs.Dirs.fairseq_models.glob('*'):
        for checkpoint_path in path_model_data.glob(f'checkpoint_last+rep=*.pt'):

            if 'AO-CHILDES' in path_model_data.name:
                bin_name = 'aochildes-data-bin'
                corpora = 'AO-CHILDES'
            elif 'Wikipedia-1' in path_model_data.name:
                bin_name = 'wikipedia1_new1_seg'
                corpora = 'Wikipedia-1'
            else:
                raise AttributeError('Invalid model name.')

            model_fairseq = RobertaModel.from_pretrained(model_name_or_path=str(path_model_data),
                                                         checkpoint_file=str(path_model_data / checkpoint_path.name),
                                                         data_name_or_path=str(path_model_data / bin_name),
                                                         )
            model, tokenizer = convert_fairseq_roberta_to_pytorch(model_fairseq)

            # collect
            rep = checkpoint_path.stem.split('+')[-1]
            model_data = ModelData(name='RoBERTa-base',
                                   corpora=corpora,
                                   rep=rep,
                                   model=model,
                                   tokenizer=tokenizer,
                                   case_sensitive=True,
                                   trained_on_lower_cased_data=True,
                                   )
            res.append(model_data)

    return res


def load_roberta_base_models():

    # pre-trained by others
    huggingface_models = [ModelData(name='RoBERTa-base',
                                    corpora='Warstadt2020',
                                    rep='0',
                                    model=AutoModelForMaskedLM.from_pretrained("nyu-mll/roberta-base-10M-1"),
                                    tokenizer=AutoTokenizer.from_pretrained("nyu-mll/roberta-base-10M-1"),
                                    case_sensitive=True,
                                    trained_on_lower_cased_data=False,
                                    ),
                          ModelData(name='RoBERTa-base',
                                    corpora='Warstadt2020',
                                    rep='1',
                                    model=AutoModelForMaskedLM.from_pretrained("nyu-mll/roberta-base-10M-2"),
                                    tokenizer=AutoTokenizer.from_pretrained("nyu-mll/roberta-base-10M-2"),
                                    case_sensitive=True,
                                    trained_on_lower_cased_data=False,
                                    ),
                          ModelData(name='RoBERTa-base',
                                    corpora='Warstadt2020',
                                    rep='2',
                                    model=AutoModelForMaskedLM.from_pretrained("nyu-mll/roberta-base-10M-3"),
                                    tokenizer=AutoTokenizer.from_pretrained("nyu-mll/roberta-base-10M-3"),
                                    case_sensitive=True,
                                    trained_on_lower_cased_data=False,
                                    ),
                          ModelData(name='RoBERTa-base',
                                    corpora='Liu2019',
                                    rep='0',
                                    model=RobertaForMaskedLM.from_pretrained('roberta-base'),
                                    tokenizer=RobertaTokenizerFast.from_pretrained('roberta-base'),
                                    case_sensitive=True,
                                    trained_on_lower_cased_data=False,
                                    )]

    # trained by us using fairseq - need to be converted
    fairseq_models = load_fairseq_as_huggingface_models()

    res = []
    res.extend(huggingface_models)
    res.extend(fairseq_models)
    return res
