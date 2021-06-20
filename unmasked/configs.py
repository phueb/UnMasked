from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    test_suites = root / 'test_suites'
    fairseq_models = root / 'fairseq_models'
    results = root / 'results'
    babyberta_runs = Path('/') / 'media' / 'ludwig_data' / 'BabyBERTa' / 'runs'  # restricted access


class Scoring:
    batch_size = 200
