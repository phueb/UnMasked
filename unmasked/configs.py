from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    test_suites = root / 'test_suites'
    fairseq_models = root / 'fairseq_models'
    results = root / 'results'
    babyberta_runs = Path('/') / 'media' / 'ludwig_data' / 'BabyBERTa' / 'runs'  # restricted access


class Scoring:
    batch_size = 10  # needs to be small to probe roberta-base


class Figs:
    lw = 1
    ax_font_size = 6
    leg_font_size = 6
    title_font_size = 6
    tick_font_size = 6
    legend_offset_from_bottom = 0.15
