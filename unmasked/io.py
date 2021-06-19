import yaml
from typing import List, Dict, Any
from pathlib import Path


def save_forced_choice_predictions(raw_sentences: List[str],
                                   cross_entropies: List[float],
                                   out_path: Path,
                                   verbose: bool = False,
                                   ) -> None:
    print(f'Saving forced_choice probing results to {out_path}')
    with out_path.open('w') as f:
        for s, xe in zip(raw_sentences, cross_entropies):
            line = f'{s} {xe:.4f}'
            f.write(line + '\n')
            if verbose:
                print(line)


def save_yaml_file(path_out: Path,
                   param2val: Dict[str, Any],
                   ):
    if not path_out.parent.exists():
        path_out.parent.mkdir()
    with path_out.open('w', encoding='utf8') as f:
        yaml.dump(param2val, f, default_flow_style=False, allow_unicode=True)


