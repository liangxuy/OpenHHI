import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader

from models.vq.unified_model import UnifiedRVQVAE
from models.vq.vq_trainer_uni import UnifiedRVQTokenizerTrainer
from options.vq_option import arg_parse
from utils.text_tokenizer import CaptionTokenizer

os.environ["OMP_NUM_THREADS"] = "1"


INTERX_REQUIRES_PROCESSED_LOADER = {'noncanonical', 'axis', 'quaternion', 'matrix', 'joint'}


def maybe_load_pretrained_vq(model, ckpt_path, device):
    if not ckpt_path:
        print('Training unified VQ without a pretrained VQ initialization.')
        return False
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'VQ initialization checkpoint not found: {ckpt_path}. '
            'Download vq_rot6d/model/finest.tar as described in README.md, '
            'or pass --init_vq_ckpt with the path to the downloaded checkpoint.'
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('vq_model', checkpoint.get('net', checkpoint))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    missing_encoder_keys = [key for key in missing_keys if key.startswith('encoder.')]
    if missing_encoder_keys:
        raise RuntimeError(
            f'VQ initialization checkpoint `{ckpt_path}` does not contain all encoder weights.\n'
            f'Missing encoder keys: {missing_encoder_keys}'
        )

    print(f'Loaded pretrained VQ from {ckpt_path}')
    print(f'Missing keys: {missing_keys}')
    print(f'Unexpected keys: {unexpected_keys}')
    return True


def freeze_pretrained_encoder(model):
    frozen_params = 0
    for param in model.encoder.parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    print(f'Froze VQ encoder parameters: {frozen_params}')


if __name__ == "__main__":
    opt = arg_parse(True)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.anim_dir = pjoin(opt.save_root, 'animation')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.log_dir = pjoin(opt.save_root, 'log')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.anim_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name != "interx":
        raise NotImplementedError("train_vq_uni.py currently supports Inter-X only.")

    opt.data_rep = str(getattr(opt, 'data_rep', 'rot6d')).lower()
    if opt.data_rep in INTERX_REQUIRES_PROCESSED_LOADER and not opt.use_processed_loader:
        raise ValueError(
            f'`--data_rep {opt.data_rep}` requires `--use_processed_loader` '
            f'with preprocessed files under data/Inter-X_Dataset/processed/motions_{opt.data_rep}.'
        )

    opt.data_root = 'data/Inter-X_Dataset'
    opt.text_dir = pjoin(opt.data_root, 'processed/texts_processed')
    opt.motion_rep = "smpl"
    opt.joints_num = 56
    opt.dim_joint = 12 if opt.data_rep == 'noncanonical' else 6
    opt.max_motion_length = 150
    opt.max_text_len = 35
    opt.unit_length = 4

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

    if opt.use_processed_loader:
        from data.interx import MotionCaptionDatasetV2HHIProcessed as MotionCaptionDatasetClass
    else:
        from data.interx import MotionCaptionDatasetV2HHI as MotionCaptionDatasetClass

    motion_train_path = pjoin(opt.motion_dir, 'train.h5')
    motion_val_path = pjoin(opt.motion_dir, 'val.h5')
    split_train = pjoin(opt.data_root, 'splits/train.txt')
    split_val = pjoin(opt.data_root, 'splits/val.txt')

    tokenizer = CaptionTokenizer.from_split_files(opt.text_dir, [split_train, split_val])
    print(f'Caption tokenizer vocab size: {len(tokenizer)}')

    dataset_kwargs = {'data_rep': opt.data_rep} if opt.use_processed_loader else {}
    train_dataset = MotionCaptionDatasetClass(opt, split_train, motion_train_path, tokenizer, **dataset_kwargs)
    val_dataset = MotionCaptionDatasetClass(opt, split_val, motion_val_path, tokenizer, **dataset_kwargs)

    if opt.use_processed_loader:
        sample_motion, _, _, _, _ = train_dataset[0]
        if opt.data_rep == 'noncanonical':
            if sample_motion.ndim != 2:
                raise ValueError(f"Expected noncanonical sample [T, 664], got {sample_motion.shape}")
            opt.joints_num = 56
            opt.dim_joint = 12
        elif sample_motion.ndim == 3:
            opt.joints_num = sample_motion.shape[1]
            opt.dim_joint = sample_motion.shape[2]
        elif sample_motion.ndim == 2:
            opt.joints_num = 1
            opt.dim_joint = sample_motion.shape[1]
        else:
            raise ValueError(f"Unsupported sample shape from processed loader: {sample_motion.shape}")

    net = UnifiedRVQVAE(
        opt,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_id,
        input_dim=opt.dim_joint,
        nb_code=opt.nb_code,
        code_dim=opt.code_dim,
        output_emb_width=opt.code_dim,
        down_t=opt.down_t,
        stride_t=opt.stride_t,
        width=opt.width,
        depth=opt.depth,
        dilation_growth_rate=opt.dilation_growth_rate,
        activation=opt.vq_act,
        norm=opt.vq_norm,
    )

    loaded_pretrained_vq = maybe_load_pretrained_vq(net, opt.init_vq_ckpt, opt.device)
    if loaded_pretrained_vq:
        freeze_pretrained_encoder(net)

    pc_vq = sum(param.numel() for param in net.parameters())
    pc_vq_enc = sum(param.numel() for param in net.encoder.parameters())
    pc_vq_dec = sum(param.numel() for param in net.decoder.parameters())
    print(net)
    print('Total parameters of Unified VQVAE: {}M'.format(pc_vq / 1000_000))
    print('Total parameters of encoder: {}M'.format(pc_vq_enc / 1000_000))
    print('Total parameters of decoder: {}M'.format(pc_vq_dec / 1000_000))

    trainer = UnifiedRVQTokenizerTrainer(opt, vq_model=net)

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

    trainer.train(train_loader, val_loader)
