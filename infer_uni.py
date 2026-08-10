import os
import time
import numpy as np
import torch
from os.path import join as pjoin
from tqdm import tqdm

from options.eval_option import arg_parse
from utils.unified_vq_loader import ensure_unified_vq_opt, load_unified_vq_model
from models.mask_transformer.transformer import MaskTransformer

from utils.get_opt import get_opt
from utils.interx_text import apply_interx_text_config
from utils.utils import fixseed
from data.utils import MotionNormalizer
import data.rotation_conversions as geometry

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')


def _resolve_text_source_override(opt):
    text_source = getattr(opt, 'text_source', None)
    if text_source is None:
        return None
    text_source = str(text_source).strip()
    return text_source or None

INTERX_REP_TO_DIM = {
    'rot6d': 6,
    'canonical': 6,
    'noncanonical': 12,
    'axis': 3,
    'quaternion': 4,
    'matrix': 9,
}
INTERX_SMPLX_REPS = tuple(INTERX_REP_TO_DIM.keys())

NONCANONICAL_JOINTS = 55
NONCANONICAL_POS_DIM = NONCANONICAL_JOINTS * 3
NONCANONICAL_VEL_DIM = NONCANONICAL_JOINTS * 3
NONCANONICAL_ROT_DIM = NONCANONICAL_JOINTS * 6
NONCANONICAL_FC_DIM = 4
NONCANONICAL_PERSON_DIM = NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM + NONCANONICAL_ROT_DIM + NONCANONICAL_FC_DIM


def _list_tar_checkpoints(model_dir):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'Model directory not found: {model_dir}')
    return sorted([f for f in os.listdir(model_dir) if f.endswith('.tar')])


def _resolve_single_checkpoint(model_dir, requested=None, fallback_priority=None):
    files = _list_tar_checkpoints(model_dir)
    fallback_priority = [] if fallback_priority is None else list(fallback_priority)

    if requested and requested != 'all':
        req_candidates = [requested]
        if not requested.endswith('.tar'):
            req_candidates.append(f'{requested}.tar')
        for candidate in req_candidates:
            if candidate in files:
                return candidate
        contains = [f for f in files if requested in f]
        if contains:
            return contains[0]
        print(f"[Warning] Requested checkpoint `{requested}` not found in {model_dir}. Falling back to priority list.")

    for candidate in fallback_priority:
        if candidate in files:
            return candidate

    return files[0]


def _resolve_trans_checkpoints(model_dir, which_epoch):
    files = _list_tar_checkpoints(model_dir)
    if which_epoch == 'all':
        return files
    selected = _resolve_single_checkpoint(
        model_dir,
        requested=which_epoch,
        fallback_priority=['finest.tar', 'best_fid.tar', 'best_acc.tar', 'latest.tar'],
    )
    return [selected]


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


def _motion_to_smplx_params(motion, data_rep):
    """
    Convert one-person generated motion from representation `data_rep` to a
    stable SMPL-X parameter dict payload.
    """
    data_rep = str(data_rep).lower()
    if data_rep not in INTERX_SMPLX_REPS:
        raise ValueError(
            f'Unsupported interx data_rep `{data_rep}` for SMPL-X export. '
            f'Supported: {INTERX_SMPLX_REPS}'
        )

    if data_rep == 'noncanonical' and motion.ndim == 3 and motion.shape[-1] == NONCANONICAL_PERSON_DIM:
        pos = motion[..., :NONCANONICAL_POS_DIM].reshape(motion.shape[0], motion.shape[1], NONCANONICAL_JOINTS, 3)
        vel = motion[..., NONCANONICAL_POS_DIM:NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM].reshape(
            motion.shape[0], motion.shape[1], NONCANONICAL_JOINTS, 3
        )
        rot_block = motion[..., NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM:
                           NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM + NONCANONICAL_ROT_DIM].reshape(
            motion.shape[0], motion.shape[1], NONCANONICAL_JOINTS, 6
        )
        root_trans = pos[..., 0, :]
        root_vel = vel[..., 0, :]
        pose = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(rot_block))
    else:
        if motion.ndim != 4:
            raise ValueError(
                f"Expected motion rank 4 for `{data_rep}` (or rank 3 flat noncanonical), got {motion.shape}"
            )
        rot_block = motion[:, :, :-1, :]
        root_block = motion[:, :, -1, :]
        root_trans = root_block[..., :3]
        if root_block.shape[-1] >= 6:
            root_vel = root_block[..., 3:6]
        else:
            root_vel = torch.zeros_like(root_trans)

        if data_rep in ('rot6d', 'canonical', 'noncanonical'):
            pose = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(rot_block))
        elif data_rep == 'axis':
            pose = rot_block
        elif data_rep == 'quaternion':
            quat_norm = rot_block.norm(dim=-1, keepdim=True)
            finite_mask = torch.isfinite(rot_block).all(dim=-1, keepdim=True)
            valid_mask = finite_mask & (quat_norm > 1e-8)
            safe_norm = torch.where(valid_mask, quat_norm, torch.ones_like(quat_norm))
            quat = rot_block / safe_norm
            identity = torch.zeros_like(quat)
            identity[..., 0] = 1.0
            quat = torch.where(valid_mask.expand_as(quat), quat, identity)
            pose = geometry.quaternion_to_axis_angle(quat)
        elif data_rep == 'matrix':
            rot_mat = rot_block.view(*rot_block.shape[:-1], 3, 3)
            pose = geometry.matrix_to_axis_angle(rot_mat)
        else:
            raise ValueError(f'Unsupported interx data_rep `{data_rep}`')

    root_orient = pose[:, :, 0, :]
    body_pose = pose[:, :, 1:, :]
    return root_trans, root_vel, root_orient, body_pose


def load_vq_model(vq_opt, which_epoch):
    model_dir = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, "model")
    ckpt_file = _resolve_single_checkpoint(
        model_dir,
        requested=which_epoch,
        fallback_priority=["finest.tar", "best_fid.tar", "latest.tar"],
    )
    ckpt_path = pjoin(model_dir, ckpt_file)

    vq_model, vq_epoch, ckpt_file = load_unified_vq_model(vq_opt, ckpt_path, device=opt.device)
    print(f"Loading Unified VQ Model {vq_opt.name} from {ckpt_file} Completed!, Epoch {vq_epoch}")
    return vq_model, vq_epoch

def load_trans_model(model_opt, which_model):
    clip_version = 'checkpoints/ViT-L-14-336px.pt'
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=768,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    assert all([k.startswith('clip_') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from {which_model} epoch {ckpt["ep"]}!')
    return t2m_transformer


def gen_motions(opt, ckpt_file, texts, net, trans, motion_len=90, data_rep='rot6d'):
    normalizer = MotionNormalizer()
    preprocess_plot_motion = None
    net = net.to(opt.device)
    net.eval()
    trans = trans.to(opt.device)
    trans.eval()

    num_samples = 1
    motion_lens = torch.tensor([motion_len] * num_samples)
    ids_length = (motion_lens.detach().long().to(opt.device) // 4)
    file_prefix = f"infer_{ckpt_file.split('.')[0]}"

    for cond_scale in opt.cond_scales:
        for time_steps in opt.time_steps:
            for topkr in opt.topkr:
                time_taken = []
                for i, text in tqdm(enumerate(texts)):
                    with torch.no_grad():
                        text = [text] * num_samples
                        tick = time.time()
                        motion_ids = trans.generate(text, ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1)
                        motion_ids1 = motion_ids[:, :motion_ids.shape[1] // 2]
                        motion_ids2 = motion_ids[:, motion_ids.shape[1] // 2:]

                        motion1_output = net.forward_decoder(motion_ids1.unsqueeze_(-1).to(opt.device))
                        motion2_output = net.forward_decoder(motion_ids2.unsqueeze_(-1).to(opt.device))
                        time_taken.append(time.time() - tick)

                        if opt.dataset_name == "interhuman":
                            if preprocess_plot_motion is None:
                                from utils.plot_script import preprocess_plot_motion as preprocess_plot_motion_fn
                                preprocess_plot_motion = preprocess_plot_motion_fn
                            motion_output = torch.cat([motion1_output, motion2_output], dim=-1)
                            motions_output = motion_output.reshape(motion_output.shape[0], motion_output.shape[1], 2, -1)
                            motions_output = normalizer.backward(motions_output.cpu().detach().numpy())
                            for motion_i in range(motions_output.shape[0]):
                                gen_file_name = f"{file_prefix}_ts{time_steps}_cs{cond_scale}_topkr{topkr}_{i:02d}_{motion_i:02d}"
                                preprocess_plot_motion(motions_output[motion_i],
                                                       text[0],
                                                       opt.vis_dir,
                                                       opt.npy_dir,
                                                       gen_file_name,
                                                       foot_ik=True)
                        elif opt.dataset_name == 'interx':
                            root_trans1, root_vel1, root_orient1, body_pose1 = _motion_to_smplx_params(motion1_output, data_rep)
                            root_trans2, root_vel2, root_orient2, body_pose2 = _motion_to_smplx_params(motion2_output, data_rep)
                            B = motion1_output.shape[0]
                            for motion_i in range(B):
                                gen_file_name = f"{file_prefix}_ts{time_steps}_cs{cond_scale}_topkr{topkr}_{i:02d}_{motion_i:02d}.npy"
                                smplx_params = {
                                    'person1': {
                                        'root_trans': root_trans1[motion_i].cpu().detach().numpy(),
                                        'root_vel': root_vel1[motion_i].cpu().detach().numpy(),
                                        'root_orient': root_orient1[motion_i].cpu().detach().numpy(),
                                        'body_pose': body_pose1[motion_i].cpu().detach().numpy(),
                                    },
                                    'person2': {
                                        'root_trans': root_trans2[motion_i].cpu().detach().numpy(),
                                        'root_vel': root_vel2[motion_i].cpu().detach().numpy(),
                                        'root_orient': root_orient2[motion_i].cpu().detach().numpy(),
                                        'body_pose': body_pose2[motion_i].cpu().detach().numpy(),
                                    }
                                }
                                np.save(pjoin(opt.vis_dir, gen_file_name), smplx_params, allow_pickle=True)
                        else:
                            raise KeyError('Dataset Does not Exists')
                print(f"Avg Time taken: {np.mean(time_taken)} secs")


if __name__ == '__main__':
    opt = arg_parse()
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    if not opt.use_trans:
        raise ValueError('infer.py currently supports text generation only with `--use_trans True`.')

    trans_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
    main_opt = get_opt(trans_opt_path, opt.device)
    fixseed(getattr(main_opt, 'seed', 3407))
    text_source_override = _resolve_text_source_override(opt)

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, main_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, opt.device)
    ensure_unified_vq_opt(vq_opt, getattr(main_opt, "vq_name", None))

    main_opt.num_tokens = vq_opt.nb_code
    main_opt.code_dim = vq_opt.code_dim

    if main_opt.dataset_name == "interhuman":
        main_opt.data_root = 'data/InterHuman'
        main_opt.joints_num = 22
        main_opt.dim_joint = 12
        main_opt.data_rep = 'rot6d'
    elif main_opt.dataset_name == "interx":
        main_opt.data_root = 'data/Inter-X_Dataset'
        main_opt.max_motion_length = 150
        apply_interx_text_config(main_opt, text_source=text_source_override, require_exists=False)
        main_opt.unit_length = 4
        main_opt.motion_rep = "smpl"
        main_opt.data_rep = str(getattr(main_opt, 'data_rep', getattr(vq_opt, 'data_rep', 'rot6d'))).lower()
        if main_opt.data_rep not in INTERX_SMPLX_REPS:
            raise ValueError(
                f'Unsupported interx data_rep `{main_opt.data_rep}` for infer SMPL-X export. '
                f'Supported: {INTERX_SMPLX_REPS}'
            )
    else:
        raise KeyError('Dataset Does not Exists')

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')

    if opt.dataset_name == "interhuman":
        main_dir = 'animation_infer'
        opt.vis_dir = pjoin(opt.save_root, main_dir, 'keypoint_mp4')
        opt.npy_dir = pjoin(opt.save_root, main_dir, 'keypoint_npy')
        os.makedirs(opt.vis_dir, exist_ok=True)
        os.makedirs(opt.npy_dir, exist_ok=True)
    elif opt.dataset_name == "interx":
        opt.vis_dir = pjoin(opt.save_root, 'animation_infer', 'smpl_npy')
        os.makedirs(opt.vis_dir, exist_ok=True)

    with open("./prompts.txt") as f:
        texts = [line.strip("\n") for line in f.readlines()]
    print(texts)

    trans_checkpoints = _resolve_trans_checkpoints(opt.model_dir, opt.which_epoch)
    vq_ckpt_request = "finest.tar"
    for trans_ckpt in trans_checkpoints:
        print(f"\n\nLoading model epoch: {trans_ckpt}")
        trans = load_trans_model(main_opt, trans_ckpt)
        net, _ = load_vq_model(vq_opt, vq_ckpt_request)
        gen_motions(opt, trans_ckpt, texts, net, trans, data_rep=getattr(main_opt, 'data_rep', 'rot6d'))
