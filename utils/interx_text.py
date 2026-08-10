import os
from os.path import join as pjoin


INTERX_TEXT_SOURCE_SPECS = {
    'original': ('texts_processed', 35),
    'simplified': ('simplified_texts_processed', 20),
    'simplist': ('simplist_texts_processed', 15),
}

INTERX_TEXT_GLOVE_SUBDIRS = {
    'original': 'glove',
    'simplified': 'simplified_glove',
    'simplist': 'simplist_glove',
}

INTERX_TEXT_EVAL_MODEL_NAMES = {
    'original': 'text_mot_match',
    'simplified': 'text_mot_match_simplified',
    'simplist': 'text_mot_match_simplist',
}

INTERX_TEXT_SOURCE_ALIASES = {
    '': 'original',
    'default': 'original',
    'orig': 'original',
    'processed': 'original',
    'texts_processed': 'original',
    'simplified_texts_processed': 'simplified',
    'simplist_texts_processed': 'simplist',
}


def normalize_interx_text_source(text_source):
    if text_source is None:
        return 'original'

    normalized = str(text_source).strip().lower()
    normalized = INTERX_TEXT_SOURCE_ALIASES.get(normalized, normalized)
    if normalized not in INTERX_TEXT_SOURCE_SPECS:
        valid_sources = ', '.join(sorted(INTERX_TEXT_SOURCE_SPECS.keys()))
        raise ValueError(f"Unsupported Inter-X text_source `{text_source}`. Expected one of: {valid_sources}")
    return normalized


def resolve_interx_text_config(data_root, text_source=None, require_exists=False):
    normalized = normalize_interx_text_source(text_source)
    text_subdir, max_text_len = INTERX_TEXT_SOURCE_SPECS[normalized]
    text_dir = pjoin(data_root, 'processed', text_subdir)

    if require_exists and not os.path.isdir(text_dir):
        raise FileNotFoundError(
            f"Inter-X text directory not found for text_source={normalized}: {text_dir}"
        )

    return normalized, text_dir, max_text_len


def resolve_interx_eval_model_name(text_source=None):
    normalized = normalize_interx_text_source(text_source)
    return INTERX_TEXT_EVAL_MODEL_NAMES[normalized]


def resolve_interx_glove_dir(data_root, text_source=None, require_exists=False):
    normalized = normalize_interx_text_source(text_source)
    glove_subdir = INTERX_TEXT_GLOVE_SUBDIRS[normalized]
    glove_dir = pjoin(data_root, 'processed', glove_subdir)

    if require_exists and not os.path.isdir(glove_dir):
        raise FileNotFoundError(
            f"Inter-X glove directory not found for text_source={normalized}: {glove_dir}"
        )

    return normalized, glove_dir


def apply_interx_text_config(opt, data_root=None, text_source=None, require_exists=False):
    data_root = getattr(opt, 'data_root', None) if data_root is None else data_root
    if data_root is None:
        raise ValueError('Inter-X text config requires a valid data_root.')

    requested_source = getattr(opt, 'text_source', None) if text_source is None else text_source
    normalized, text_dir, max_text_len = resolve_interx_text_config(
        data_root,
        requested_source,
        require_exists=require_exists,
    )
    _, glove_dir = resolve_interx_glove_dir(
        data_root,
        normalized,
        require_exists=require_exists,
    )

    opt.text_source = normalized
    opt.text_dir = text_dir
    opt.glove_dir = glove_dir
    opt.max_text_len = max_text_len
    return opt
