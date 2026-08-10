import os
import random
import time
from os.path import join as pjoin

import numpy as np
import torch
from tqdm import tqdm

import infer_uni as infer_base
from data.interx import _load_text_data
from options.eval_option import arg_parse
from utils.get_opt import get_opt
from utils.interx_text import apply_interx_text_config
from utils.unified_vq_loader import ensure_unified_vq_opt
from utils.utils import fixseed


NUM_PROMPTS = 200
RESULT_ROOT = './outputs/openhhi_random_infer'


def _as_list(value):
    """Normalize argparse values that may be either a scalar or a list."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def sample_test_captions(data_root, text_dir, num_prompts, seed):
    """Sample full-motion captions from the Inter-X test split."""
    split_file = pjoin(data_root, 'processed', 'splits', 'test.txt')
    if not os.path.isfile(split_file):
        split_file = pjoin(data_root, 'splits', 'test.txt')
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f'Inter-X test split not found under {data_root}')

    with open(split_file, 'r', encoding='utf-8') as f:
        test_ids = [line.strip() for line in f if line.strip()]

    caption_pool = []
    for sample_id in test_ids:
        for text_data in _load_text_data(text_dir, sample_id):
            caption_pool.append((sample_id, text_data['caption']))

    if num_prompts > len(caption_pool):
        raise ValueError(
            f'Requested {num_prompts} prompts, but the Inter-X test split only '
            f'contains {len(caption_pool)} valid full-motion captions.'
        )

    rng = random.Random(seed)
    sampled = rng.sample(caption_pool, num_prompts)
    sample_ids = [sample_id for sample_id, _ in sampled]
    captions = [caption for _, caption in sampled]
    return captions, sample_ids, split_file, len(caption_pool)


def save_prompts(prompts, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    prompt_file = pjoin(result_dir, 'prompts.txt')
    with open(prompt_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(prompt.rstrip('\n') + '\n')
    return prompt_file


def save_sample_ids(sample_ids, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    id_file = pjoin(result_dir, 'id.txt')
    with open(id_file, 'w', encoding='utf-8') as f:
        for sample_id in sample_ids:
            f.write(sample_id.rstrip('\n') + '\n')
    return id_file


def make_unique_vis_dir(result_root, exp_name):
    run_root = pjoin(result_root, exp_name, 'runs')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    for suffix in range(1000):
        run_name = timestamp if suffix == 0 else f'{timestamp}_{suffix:03d}'
        run_dir = pjoin(run_root, run_name)
        try:
            os.makedirs(run_dir, exist_ok=False)
        except FileExistsError:
            continue

        vis_dir = pjoin(run_dir, 'smpl_npy')
        os.makedirs(vis_dir, exist_ok=False)
        return vis_dir

    raise RuntimeError(f'Failed to create a unique output directory under {run_root}')


def gen_motions(opt, texts, net, trans, motion_len=90, data_rep='rot6d'):
    """Generate one motion per prompt and save it as <prompt_line_id>.npy."""
    net = net.to(opt.device)
    net.eval()
    trans = trans.to(opt.device)
    trans.eval()

    cond_scales = _as_list(opt.cond_scales)
    time_steps_values = _as_list(opt.time_steps)
    topkr_values = _as_list(opt.topkr)
    num_settings = len(cond_scales) * len(time_steps_values) * len(topkr_values)
    if num_settings != 1:
        raise ValueError(
            'Exactly one cond_scale, time_steps, and topkr is required because '
            'ID-only output names would otherwise overwrite each other.'
        )

    cond_scale = cond_scales[0]
    time_steps = time_steps_values[0]
    topkr = topkr_values[0]
    motion_lens = torch.tensor([motion_len])
    ids_length = motion_lens.detach().long().to(opt.device) // 4
    time_taken = []

    for prompt_id, text in tqdm(
        enumerate(texts), total=len(texts), desc='Generating motions'
    ):
        with torch.no_grad():
            tick = time.time()
            motion_ids = trans.generate(
                [text],
                ids_length,
                time_steps,
                cond_scale,
                topk_filter_thres=topkr,
                temperature=opt.temperature,
            )
            split_idx = motion_ids.shape[1] // 2
            motion_ids1 = motion_ids[:, :split_idx]
            motion_ids2 = motion_ids[:, split_idx:]

            motion1_output = net.forward_decoder(motion_ids1.unsqueeze(-1).to(opt.device))
            motion2_output = net.forward_decoder(motion_ids2.unsqueeze(-1).to(opt.device))
            time_taken.append(time.time() - tick)

            root_trans1, root_vel1, root_orient1, body_pose1 = infer_base._motion_to_smplx_params(
                motion1_output, data_rep
            )
            root_trans2, root_vel2, root_orient2, body_pose2 = infer_base._motion_to_smplx_params(
                motion2_output, data_rep
            )
            smplx_params = {
                'person1': {
                    'root_trans': root_trans1[0].cpu().numpy(),
                    'root_vel': root_vel1[0].cpu().numpy(),
                    'root_orient': root_orient1[0].cpu().numpy(),
                    'body_pose': body_pose1[0].cpu().numpy(),
                },
                'person2': {
                    'root_trans': root_trans2[0].cpu().numpy(),
                    'root_vel': root_vel2[0].cpu().numpy(),
                    'root_orient': root_orient2[0].cpu().numpy(),
                    'body_pose': body_pose2[0].cpu().numpy(),
                },
            }
            np.save(pjoin(opt.vis_dir, f'{prompt_id}.npy'), smplx_params, allow_pickle=True)

    print(f'Average generation time: {np.mean(time_taken):.4f} seconds')


def main():
    opt = arg_parse()
    if opt.dataset_name != 'interx':
        raise ValueError('infer_uni_random.py only supports --dataset_name interx.')
    if not opt.use_trans:
        raise ValueError('infer_uni_random.py requires --use_trans True.')

    opt.device = torch.device('cpu' if opt.gpu_id == -1 else f'cuda:{opt.gpu_id}')
    print(f'Using Device: {opt.device}')

    trans_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    main_opt = get_opt(trans_opt_path, opt.device)
    seed = getattr(main_opt, 'seed', 3407)
    fixseed(seed)

    text_source_override = infer_base._resolve_text_source_override(opt)
    main_opt.data_root = 'data/Inter-X_Dataset'
    main_opt.max_motion_length = 150
    apply_interx_text_config(
        main_opt, text_source=text_source_override, require_exists=True
    )
    main_opt.unit_length = 4
    main_opt.motion_rep = 'smpl'

    vq_opt_path = pjoin(
        opt.checkpoints_dir, opt.dataset_name, main_opt.vq_name, 'opt.txt'
    )
    vq_opt = get_opt(vq_opt_path, opt.device)
    ensure_unified_vq_opt(vq_opt, getattr(main_opt, 'vq_name', None))
    main_opt.num_tokens = vq_opt.nb_code
    main_opt.code_dim = vq_opt.code_dim
    main_opt.data_rep = str(
        getattr(main_opt, 'data_rep', getattr(vq_opt, 'data_rep', 'rot6d'))
    ).lower()
    if main_opt.data_rep not in infer_base.INTERX_SMPLX_REPS:
        raise ValueError(
            f'Unsupported interx data_rep `{main_opt.data_rep}` for SMPL-X export. '
            f'Supported: {infer_base.INTERX_SMPLX_REPS}'
        )

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.vis_dir = make_unique_vis_dir(RESULT_ROOT, opt.name)
    print(f'Created output directory: {opt.vis_dir}')

    texts, sample_ids, split_file, pool_size = sample_test_captions(
        main_opt.data_root, main_opt.text_dir, NUM_PROMPTS, seed
    )
    prompt_file = save_prompts(texts, opt.vis_dir)
    id_file = save_sample_ids(sample_ids, opt.vis_dir)
    print(
        f'Sampled {len(texts)} of {pool_size} valid captions from {split_file} '
        f'with seed {seed}.'
    )
    print(f'Saved prompts to {prompt_file}')
    print(f'Saved sample ids to {id_file}')

    trans_checkpoints = infer_base._resolve_trans_checkpoints(
        opt.model_dir, opt.which_epoch
    )
    if len(trans_checkpoints) != 1:
        raise ValueError(
            'Exactly one transformer checkpoint is required because ID-only output '
            'names would otherwise overwrite each other.'
        )

    # The imported model-loading helpers use infer_uni.opt as their runtime context.
    infer_base.opt = opt
    trans_ckpt = trans_checkpoints[0]
    print(f'Loading model epoch: {trans_ckpt}')
    trans = infer_base.load_trans_model(main_opt, trans_ckpt)
    net, _ = infer_base.load_vq_model(vq_opt, 'finest.tar')
    gen_motions(opt, texts, net, trans, data_rep=main_opt.data_rep)


if __name__ == '__main__':
    main()
