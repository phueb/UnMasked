"""
Load all models. 
There are three different classes of models:
1. BabyBERTa trained with huggingface transformers by us. (loaded from a restricted remote drive)
2. RoBERTa-base trained in fairseq by us. (loaded via checkpoint files)
3. RoBERTa-base trained by others. (loaded using huggingface transformers API)

Notes:
    - We do not load all models at once - this wastes CUDA memory.
    Instead we specify directions for how load each model using functools.partial.
"""

import yaml
from typing import List
from dataclasses import dataclass
from functools import partial

from fairseq.models.roberta import RobertaModel
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import AutoTokenizer, AutoModelForMaskedLM

from unmasked import configs
from unmasked.convert_fairseq_roberta import convert_fairseq_roberta_to_pytorch


@dataclass
class ModelData:
    name: str
    corpora: str
    rep: int
    model: partial
    tokenizer: partial
    path: str

    # use these two attributes to decide if to evaluate model on lower-cased test sentences
    case_sensitive: bool
    trained_on_lower_cased_data: bool

    # specific to babyberta models (to trace each replication back to a unique location on the shared drive)


def load_babyberta_models(param_numbers: List[int],
                          ) -> List[ModelData]:
    """
    load all BabyBERTa models trained by us.

    notes:
    - models are stored on shared remote drive with restricted access.
    - there are multiple replications of each model (with different initializations)

    - load models in alphabetical order of their path to assign "rep" consistently
    """
    res = []
    for param_name in [f'param_{i:03}' for i in param_numbers]:
        for rep, path_to_babyberta in enumerate(sorted((configs.Dirs.babyberta_runs / param_name).glob('**/saves/'))):
            with (configs.Dirs.babyberta_runs / param_name / 'param2val.yaml').open('r') as f:
                param2val = yaml.load(f, Loader=yaml.FullLoader)
            if param2val['corpora'] == ('aochildes',):
                corpora = 'AO-CHILDES'
            elif param2val['corpora'] == ('wikipedia1',):
                corpora = 'Wikipedia-1'
            elif param2val['corpora'] == ('aonewsela',):
                corpora = 'AO-Newsela'
            elif param2val['corpora'] == ('aochildes', 'aonewsela', 'wikipedia3'):
                corpora = 'AO-CHILDES+AO-Newsela+Wikipedia-3'
            else:
                raise NotImplementedError(param2val['corpora'])

            model_data = ModelData(name='BabyBERTa',
                                   corpora=corpora,
                                   rep=rep,
                                   model=partial(RobertaForMaskedLM.from_pretrained,
                                                 pretrained_model_name_or_path=str(path_to_babyberta)),
                                   tokenizer=partial(RobertaTokenizerFast.from_pretrained,
                                                     pretrained_model_name_or_path=str(path_to_babyberta),
                                                     add_prefix_space=True),
                                   case_sensitive=False,
                                   trained_on_lower_cased_data=True,
                                   path=str(path_to_babyberta)
                                   )
            res.append(model_data)

    return res


def load_fairseq_as_huggingface_models() -> List[ModelData]:
    """load Roberta-base models pre-trained by us, and convert to huggingface Roberta """

    def loading_function(path_model_data_,
                         bin_name_,
                         checkpoint_path_,
                         ) -> RobertaForMaskedLM:
        model_fairseq = RobertaModel.from_pretrained(model_name_or_path=str(path_model_data_),
                                                     checkpoint_file=str(path_model_data_ / checkpoint_path_.name),
                                                     data_name_or_path=str(path_model_data_ / bin_name_),
                                                     )
        return convert_fairseq_roberta_to_pytorch(model_fairseq)

    res = []
    for path_model_data in configs.Dirs.fairseq_models.glob('*'):
        for checkpoint_path in path_model_data.glob(f'checkpoint_last_*.pt'):

            if 'AO-CHILDES' in path_model_data.name:
                bin_name = 'aochildes-data-bin'
                corpora = 'AO-CHILDES'
            elif 'Wikipedia-1' in path_model_data.name:
                bin_name = 'wikipedia1_new1_seg'
                corpora = 'Wikipedia-1'
            else:
                raise AttributeError('Invalid model name.')

            # collect
            rep = checkpoint_path.stem[-1]
            model_data = ModelData(name='RoBERTa-base',
                                   corpora=corpora,
                                   rep=int(rep),
                                   model=partial(loading_function,
                                                 path_model_data_=path_model_data,
                                                 bin_name_=bin_name,
                                                 checkpoint_path_=checkpoint_path
                                                 ),
                                   tokenizer=partial(RobertaTokenizerFast.from_pretrained,
                                                     pretrained_model_name_or_path='roberta-base'),
                                   case_sensitive=True,
                                   trained_on_lower_cased_data=True,
                                   path=str(checkpoint_path),
                                   )
            res.append(model_data)

    return res


def load_roberta_base_models():

    # pre-trained by others
    huggingface_models = [
        ModelData(name='RoBERTa-base',
                  corpora='Warstadt2020',
                  rep=0,
                  model=partial(AutoModelForMaskedLM.from_pretrained,
                                pretrained_model_name_or_path="nyu-mll/roberta-base-10M-1"),
                  tokenizer=partial(AutoTokenizer.from_pretrained,
                                    pretrained_model_name_or_path="nyu-mll/roberta-base-10M-1"),
                  case_sensitive=True,
                  trained_on_lower_cased_data=False,
                  path="nyu-mll/roberta-base-10M-1",
                  ),
        ModelData(name='RoBERTa-base',
                  corpora='Warstadt2020',
                  rep=1,
                  model=partial(AutoModelForMaskedLM.from_pretrained,
                                pretrained_model_name_or_path="nyu-mll/roberta-base-10M-2"),
                  tokenizer=partial(AutoTokenizer.from_pretrained,
                                    pretrained_model_name_or_path="nyu-mll/roberta-base-10M-2"),
                  case_sensitive=True,
                  trained_on_lower_cased_data=False,
                  path="nyu-mll/roberta-base-10M-2",
                  ),
        ModelData(name='RoBERTa-base',
                  corpora='Warstadt2020',
                  rep=2,
                  model=partial(AutoModelForMaskedLM.from_pretrained,
                                pretrained_model_name_or_path="nyu-mll/roberta-base-10M-3"),
                  tokenizer=partial(AutoTokenizer.from_pretrained,
                                    pretrained_model_name_or_path="nyu-mll/roberta-base-10M-3"),
                  case_sensitive=True,
                  trained_on_lower_cased_data=False,
                  path="nyu-mll/roberta-base-10M-3",
                  ),
        ModelData(name='RoBERTa-base',
                  corpora='Liu2019',
                  rep=0,
                  model=partial(RobertaForMaskedLM.from_pretrained,
                                pretrained_model_name_or_path='roberta-base'),
                  tokenizer=partial(RobertaTokenizerFast.from_pretrained,
                                    pretrained_model_name_or_path='roberta-base'),
                  case_sensitive=True,
                  trained_on_lower_cased_data=False,
                  path='roberta-base'),
    ]

    # trained by us using fairseq - need to be converted
    fairseq_models = load_fairseq_as_huggingface_models()

    res = []
    res.extend(huggingface_models)
    res.extend(fairseq_models)
    return res
