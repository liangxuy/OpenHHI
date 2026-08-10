import os
from os.path import join as pjoin

import torch

from models.vq.unified_model import UnifiedRVQVAE
from utils.text_tokenizer import CaptionTokenizer


INTERX_REP_TO_DIM = {
    'rot6d': 6,
    'canonical': 6,
    'noncanonical': 12,
    'axis': 3,
    'quaternion': 4,
    'joint': 3,
    'matrix': 9,
}

INTERX_REP_TO_EXTERNAL_PERSON_DIM = {
    'rot6d': 6,
    'canonical': 6,
    'noncanonical': 664,
    'axis': 3,
    'quaternion': 4,
    'joint': 3,
    'matrix': 9,
}

INTERX_REQUIRES_PROCESSED_LOADER = {'noncanonical', 'axis', 'quaternion', 'matrix', 'joint'}

REQUIRED_UNIFIED_FIELDS = (
    'caption_num_layers',
    'uni_transformer_layers',
    'uni_transformer_heads',
    'uni_transformer_ff_size',
    'uni_dropout',
)


def ensure_unified_vq_opt(vq_opt, vq_name=None):
    if getattr(vq_opt, 'dataset_name', None) != 'interx':
        raise NotImplementedError('Unified VQ loading is currently supported for Inter-X only.')

    missing_fields = [field for field in REQUIRED_UNIFIED_FIELDS if not hasattr(vq_opt, field)]
    if missing_fields:
        display_name = vq_name or getattr(vq_opt, 'name', '<unknown>')
        raise ValueError(
            f'VQ experiment `{display_name}` does not look like a unified VQ checkpoint. '
            f'Missing opt fields: {missing_fields}'
        )

    vq_opt.data_rep = str(getattr(vq_opt, 'data_rep', 'rot6d')).lower()
    return vq_opt


def infer_interx_dim_joint(vq_opt, model_state_dict):
    dim_joint = getattr(vq_opt, 'dim_joint', None)
    if isinstance(dim_joint, int) and dim_joint > 0:
        return dim_joint
    if isinstance(dim_joint, str) and dim_joint.isdigit():
        return int(dim_joint)

    data_rep = str(getattr(vq_opt, 'data_rep', 'rot6d')).lower()
    dim_from_rep = INTERX_REP_TO_DIM.get(data_rep)

    conv_key = next((k for k in model_state_dict.keys() if k.endswith('encoder.model.0.weight')), None)
    dim_from_ckpt = None
    if conv_key is not None and model_state_dict[conv_key].ndim >= 2:
        dim_from_ckpt = int(model_state_dict[conv_key].shape[1])

    if dim_from_ckpt is not None:
        if dim_from_rep is not None and dim_from_rep != dim_from_ckpt:
            print(
                f"[Warning] VQ data_rep={data_rep} suggests dim_joint={dim_from_rep}, "
                f"but checkpoint indicates {dim_from_ckpt}. Using checkpoint value."
            )
        return dim_from_ckpt

    if dim_from_rep is not None:
        return dim_from_rep

    return 6


def build_unified_caption_tokenizer(data_root='data/Inter-X_Dataset'):
    text_dir = pjoin(data_root, 'processed/texts_processed')
    split_train = pjoin(data_root, 'splits/train.txt')
    split_val = pjoin(data_root, 'splits/val.txt')

    missing_paths = [path for path in (text_dir, split_train, split_val) if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(
            'Unified VQ reconstruction requires the same Inter-X text metadata used in stage 1. '
            f'Missing: {missing_paths}'
        )

    tokenizer = CaptionTokenizer.from_split_files(text_dir, [split_train, split_val])
    return tokenizer


def resolve_vq_checkpoint(vq_opt):
    model_dir = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model')
    ckpt_candidates = ['best_fid.tar', 'finest.tar', 'best_acc.tar', 'best_top1.tar', 'latest.tar']
    for ckpt_name in ckpt_candidates:
        ckpt_path = pjoin(model_dir, ckpt_name)
        if os.path.isfile(ckpt_path):
            return ckpt_path

    available_files = sorted(os.listdir(model_dir)) if os.path.isdir(model_dir) else []
    raise FileNotFoundError(
        f'No VQ checkpoint found under {model_dir}. Tried {ckpt_candidates}, found {available_files}'
    )


def resolve_interx_motion_dir(data_root, use_processed_loader, data_rep):
    default_motion_dir = pjoin(data_root, 'processed/motions')
    if not use_processed_loader:
        return default_motion_dir

    candidate_motion_dir = pjoin(data_root, f'processed/motions_{data_rep}')
    if os.path.isdir(candidate_motion_dir):
        return candidate_motion_dir

    if data_rep in INTERX_REQUIRES_PROCESSED_LOADER:
        raise FileNotFoundError(
            f'Required processed motion dir not found for data_rep={data_rep}: {candidate_motion_dir}'
        )

    print(f'[Warning] Processed motion dir {candidate_motion_dir} not found. Falling back to {default_motion_dir}.')
    return default_motion_dir


def load_unified_vq_model(vq_opt, ckpt_path, device='cpu', strict=True):
    ensure_unified_vq_opt(vq_opt)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in checkpoint else 'net'
    model_state_dict = checkpoint[model_key]

    vq_opt.dim_joint = infer_interx_dim_joint(vq_opt, model_state_dict)
    tokenizer = build_unified_caption_tokenizer()

    vq_model = UnifiedRVQVAE(
        vq_opt,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_id,
        input_dim=vq_opt.dim_joint,
        nb_code=vq_opt.nb_code,
        code_dim=vq_opt.code_dim,
        output_emb_width=vq_opt.output_emb_width,
        down_t=vq_opt.down_t,
        stride_t=vq_opt.stride_t,
        width=vq_opt.width,
        depth=vq_opt.depth,
        dilation_growth_rate=vq_opt.dilation_growth_rate,
        activation=vq_opt.vq_act,
        norm=vq_opt.vq_norm,
    )

    missing_keys, unexpected_keys = vq_model.load_state_dict(model_state_dict, strict=False)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f'Unified VQ checkpoint `{ckpt_path}` did not load cleanly.\n'
            f'Missing keys: {missing_keys}\n'
            f'Unexpected keys: {unexpected_keys}'
        )

    vq_epoch = checkpoint['ep'] if 'ep' in checkpoint else -1
    return vq_model, vq_epoch, os.path.basename(ckpt_path)
