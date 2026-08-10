import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer
from models.vq.unified_model import UnifiedRVQVAE
from options.trans_option import TrainTransOptions

from utils.get_opt import get_opt
from utils.interx_text import (
    apply_interx_text_config,
    resolve_interx_eval_model_name,
    resolve_interx_glove_dir,
    resolve_interx_text_config,
)
from utils.text_tokenizer import CaptionTokenizer
from utils.utils import fixseed


INTERX_REP_TO_DIM = {
    'rot6d': 6,
    'canonical': 6,
    'noncanonical': 12,
    'axis': 3,
    'quaternion': 4,
    'joint': 3,
    'matrix': 9,
}
INTERX_REP_TO_EXTERNAL_DIM = {
    'rot6d': 6,
    'canonical': 6,
    'noncanonical': 664,
    'axis': 3,
    'quaternion': 4,
    'joint': 3,
    'matrix': 9,
}
INTERX_REQUIRES_PROCESSED_LOADER = {'noncanonical', 'axis', 'quaternion', 'matrix', 'joint'}


def _resolve_vq_checkpoint(vq_opt):
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


def _infer_interx_dim_joint(vq_opt, model_state_dict):
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


def _build_unified_caption_tokenizer(text_dir):
    data_root = 'data/Inter-X_Dataset'
    split_train = pjoin(data_root, 'splits/train.txt')
    split_val = pjoin(data_root, 'splits/val.txt')

    missing_paths = [path for path in (text_dir, split_train, split_val) if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(
            'Unified VQ reconstruction requires the same Inter-X text metadata used in stage 1. '
            f'Missing: {missing_paths}'
        )

    tokenizer = CaptionTokenizer.from_split_files(text_dir, [split_train, split_val])
    print(f'Unified caption tokenizer vocab size: {len(tokenizer)}')
    return tokenizer


def _resolve_pretrained_trans_checkpoint():
    if opt.pretrained_trans_ckpt:
        if not os.path.isfile(opt.pretrained_trans_ckpt):
            raise FileNotFoundError(f'Pretrained transformer checkpoint not found: {opt.pretrained_trans_ckpt}')
        return opt.pretrained_trans_ckpt

    model_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.pretrained_trans_name, 'model')
    ckpt_candidates = ['best_fid.tar', 'best_acc.tar', 'best_top1.tar', 'finest.tar', 'latest.tar']
    for ckpt_name in ckpt_candidates:
        ckpt_path = pjoin(model_dir, ckpt_name)
        if os.path.isfile(ckpt_path):
            return ckpt_path

    available_files = sorted(os.listdir(model_dir)) if os.path.isdir(model_dir) else []
    raise FileNotFoundError(
        f'No pretrained transformer checkpoint found under {model_dir}. '
        f'Tried {ckpt_candidates}, found {available_files}'
    )


def maybe_load_pretrained_transformer(mask_transformer):
    if not (opt.load_pretrained_trans or opt.pretrained_trans_ckpt):
        return

    ckpt_path = _resolve_pretrained_trans_checkpoint()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('t2m_transformer', ckpt.get('net', ckpt))
    missing_keys, unexpected_keys = mask_transformer.load_state_dict(state_dict, strict=False)

    non_clip_missing = [key for key in missing_keys if not key.startswith('clip_')]
    if unexpected_keys or non_clip_missing:
        raise RuntimeError(
            f'Pretrained transformer checkpoint `{ckpt_path}` did not load cleanly.\n'
            f'Missing keys: {missing_keys}\n'
            f'Unexpected keys: {unexpected_keys}'
        )

    print(
        f'Loaded pretrained transformer from {os.path.basename(ckpt_path)}, '
        f'epoch {ckpt.get("ep", -1)}'
    )


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)

    if vq_opt.dataset_name != "interx":
        raise NotImplementedError("train_transformer_uni.py currently supports unified Inter-X VQ only.")

    required_unified_fields = (
        'caption_num_layers',
        'uni_transformer_layers',
        'uni_transformer_heads',
        'uni_transformer_ff_size',
        'uni_dropout',
    )
    missing_unified_fields = [field for field in required_unified_fields if not hasattr(vq_opt, field)]
    if missing_unified_fields:
        raise ValueError(
            f'VQ experiment `{opt.vq_name}` does not look like a unified VQ checkpoint. '
            f'Missing opt fields: {missing_unified_fields}'
        )

    vq_opt.data_rep = str(getattr(vq_opt, 'data_rep', getattr(opt, 'data_rep', 'rot6d'))).lower()

    ckpt_path = _resolve_vq_checkpoint(vq_opt)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    model_state_dict = ckpt[model_key]

    vq_opt.dim_joint = _infer_interx_dim_joint(vq_opt, model_state_dict)
    resolved_text_source, text_dir, _ = resolve_interx_text_config(
        'data/Inter-X_Dataset',
        getattr(opt, 'text_source', None),
        require_exists=True,
    )
    opt.text_source = resolved_text_source
    tokenizer = _build_unified_caption_tokenizer(text_dir)

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
    if unexpected_keys or missing_keys:
        raise RuntimeError(
            f'Unified VQ checkpoint `{ckpt_path}` did not load cleanly.\n'
            f'Missing keys: {missing_keys}\n'
            f'Unexpected keys: {unexpected_keys}'
        )

    print(f'Loading Unified VQ Model {opt.vq_name} from {os.path.basename(ckpt_path)}, epoch {ckpt.get("ep", -1)}')
    return vq_model, vq_opt


if __name__ == '__main__':
    parser = TrainTransOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.anim_dir = pjoin(opt.save_root, 'animation')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.log_dir = pjoin(opt.save_root, 'log')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.anim_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    vq_model, vq_opt = load_vq_model()

    if opt.dataset_name == "interhuman":
        raise NotImplementedError("train_transformer_uni.py is intended for unified Inter-X VQ checkpoints.")

    if opt.dataset_name == "interx":
        opt.data_rep = str(getattr(opt, 'data_rep', getattr(vq_opt, 'data_rep', 'rot6d'))).lower()
        vq_opt.data_rep = str(getattr(vq_opt, 'data_rep', opt.data_rep)).lower()

        if opt.use_processed_loader and opt.data_rep != vq_opt.data_rep:
            raise ValueError(
                f'`--data_rep {opt.data_rep}` mismatches VQ representation `{vq_opt.data_rep}` from {opt.vq_name}.'
            )
        if not opt.use_processed_loader and vq_opt.data_rep in INTERX_REQUIRES_PROCESSED_LOADER:
            raise ValueError(
                f'VQ `{opt.vq_name}` was trained on `{vq_opt.data_rep}` features. '
                f'Use `--use_processed_loader --data_rep {vq_opt.data_rep}` for transformer training.'
            )
        if not opt.use_processed_loader and vq_opt.data_rep not in ('rot6d', 'canonical'):
            raise ValueError(
                f'VQ `{opt.vq_name}` was trained on `{vq_opt.data_rep}` features. '
                f'Use `--use_processed_loader --data_rep {vq_opt.data_rep}` for transformer training.'
            )

        opt.data_root = 'data/Inter-X_Dataset'
        default_motion_dir = pjoin(opt.data_root, 'processed/motions')
        if opt.use_processed_loader:
            candidate_motion_dir = pjoin(opt.data_root, f'processed/motions_{opt.data_rep}')
            if os.path.isdir(candidate_motion_dir):
                opt.motion_dir = candidate_motion_dir
            else:
                if opt.data_rep in INTERX_REQUIRES_PROCESSED_LOADER:
                    raise FileNotFoundError(
                        f"Required processed motion dir not found for data_rep={opt.data_rep}: {candidate_motion_dir}"
                    )
                print(f"[Warning] Processed motion dir {candidate_motion_dir} not found. Falling back to {default_motion_dir}.")
                opt.motion_dir = default_motion_dir
        else:
            opt.motion_dir = default_motion_dir
        apply_interx_text_config(opt, require_exists=True)

        opt.motion_rep = "smpl"
        opt.joints_num = 56
        opt.max_motion_length = 150
        opt.unit_length = 4

        opt.test_batch_size = 32
        opt.vq_dim_joint = int(
            INTERX_REP_TO_EXTERNAL_DIM.get(vq_opt.data_rep, getattr(vq_opt, 'dim_joint', 6))
        )
        fps = 30

        from data.interx import Text2MotionDatasetV2HHI, Text2MotionDatasetV2HHIProcessed, collate_fn
        from models.evaluator.evaluator_interx import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer

        TextDatasetClass = Text2MotionDatasetV2HHIProcessed if opt.use_processed_loader else Text2MotionDatasetV2HHI
        dataset_kwargs = {'data_rep': opt.data_rep} if opt.use_processed_loader else {}
        motion_train_path = pjoin(opt.motion_dir, 'train.h5')
        motion_val_path = pjoin(opt.motion_dir, 'val.h5')

        _, glove_dir = resolve_interx_glove_dir(opt.data_root, opt.text_source, require_exists=True)
        w_vectorizer = WordVectorizer(glove_dir, 'interx_vab')
        train_dataset = TextDatasetClass(
            opt,
            pjoin(opt.data_root, 'splits/train.txt'),
            w_vectorizer,
            motion_train_path,
            **dataset_kwargs,
        )
        val_dataset = TextDatasetClass(
            opt,
            pjoin(opt.data_root, 'splits/val.txt'),
            w_vectorizer,
            motion_val_path,
            **dataset_kwargs,
        )

        if opt.do_eval:
            test_dataset = TextDatasetClass(
                opt,
                pjoin(opt.data_root, 'splits/val.txt'),
                w_vectorizer,
                motion_val_path,
                **dataset_kwargs,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=opt.test_batch_size,
                num_workers=4,
                drop_last=True,
                collate_fn=collate_fn,
                shuffle=True,
            )

            wrapper_opt = get_opt("checkpoints/interx/text_mot_match/model/opt.txt", opt.device, complete=False)
            wrapper_opt.data_rep = opt.data_rep if opt.use_processed_loader else str(getattr(vq_opt, 'data_rep', 'rot6d')).lower()
            wrapper_opt.max_text_len = opt.max_text_len
            wrapper_opt.text_mot_match_name = resolve_interx_eval_model_name(opt.text_source)
            eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        else:
            test_loader = None
            eval_wrapper = None
    else:
        raise KeyError('Dataset Does not Exists')

    clip_version = 'checkpoints/ViT-L-14-336px.pt'
    opt.num_tokens = vq_opt.nb_code
    mask_transformer = MaskTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode='text',
        latent_dim=opt.latent_dim,
        ff_size=opt.ff_size,
        num_layers=opt.n_layers,
        num_heads=opt.n_heads,
        dropout=opt.dropout,
        clip_dim=768,
        cond_drop_prob=opt.cond_drop_prob,
        clip_version=clip_version,
        opt=opt,
    )

    pc_transformer = sum(param.numel() for param in mask_transformer.parameters_wo_clip())
    print('Total parameters of the Masked Transformer=: {:.2f}M'.format(pc_transformer / 1000_000))

    maybe_load_pretrained_transformer(mask_transformer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    opt.save_vis = False
    opt.gen_react = False

    trainer = MaskTransformerTrainer(opt, mask_transformer, vq_model)
    trainer.train(train_loader, val_loader, test_loader, eval_wrapper=eval_wrapper)
