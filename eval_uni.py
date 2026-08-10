import numpy as np
import torch
import os
from os.path import join as pjoin
from tqdm import tqdm
from datetime import datetime

from options.eval_option import arg_parse
from utils.unified_vq_loader import ensure_unified_vq_opt, load_unified_vq_model
from models.mask_transformer.transformer import MaskTransformer

from utils.metrics import *
from utils.get_opt import get_opt
from utils.interx_text import apply_interx_text_config, resolve_interx_eval_model_name
from utils.utils import fixseed
from collections import OrderedDict

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


def _log_warning(msg, file=None):
    print(msg)
    if file is not None:
        print(msg, file=file, flush=True)


def _sanitize_embedding_rows(embeddings, name, file=None, min_rows=1):
    """
    Keep only fully finite rows from embedding arrays.
    Returns a 2D array with shape [N, D].
    """
    arr = np.asarray(embeddings)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 2D embeddings, got shape {arr.shape}")

    if arr.shape[0] == 0:
        return arr

    mask = np.isfinite(arr).all(axis=1)
    removed = int((~mask).sum())
    if removed > 0:
        _log_warning(f"[Warning] {name}: dropped {removed}/{arr.shape[0]} rows with NaN/Inf.", file=file)
    arr = arr[mask]
    if arr.shape[0] < min_rows:
        _log_warning(f"[Warning] {name}: only {arr.shape[0]} valid rows remain.", file=file)
    return arr


def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                text_embeddings = text_embeddings.cpu().numpy()
                motion_embeddings = motion_embeddings.cpu().numpy()
                if text_embeddings.ndim != 2 or motion_embeddings.ndim != 2:
                    raise ValueError(
                        f"Expected 2D embeddings, got text={text_embeddings.shape}, motion={motion_embeddings.shape}"
                    )

                pair_mask = np.isfinite(text_embeddings).all(axis=1) & np.isfinite(motion_embeddings).all(axis=1)
                removed = int((~pair_mask).sum())
                if removed > 0:
                    _log_warning(
                        f"[Warning] {motion_loader_name} batch {idx}: dropped {removed}/{text_embeddings.shape[0]} pairs with NaN/Inf embeddings.",
                        file=file,
                    )
                if not pair_mask.any():
                    _log_warning(f"[Warning] {motion_loader_name} batch {idx}: no valid pairs left, skipped.", file=file)
                    continue
                text_embeddings = text_embeddings[pair_mask]
                motion_embeddings = motion_embeddings[pair_mask]

                # print(text_embeddings.shape)
                # print(motion_embeddings.shape)
                dist_mat = euclidean_distance_matrix(text_embeddings, motion_embeddings)
                diag = np.diag(dist_mat)
                if not np.isfinite(diag).all():
                    valid_idx = np.where(np.isfinite(diag))[0]
                    dropped_diag = int(diag.shape[0] - valid_idx.shape[0])
                    _log_warning(
                        f"[Warning] {motion_loader_name} batch {idx}: dropped {dropped_diag}/{diag.shape[0]} rows due to non-finite distance diagonal.",
                        file=file,
                    )
                    if valid_idx.shape[0] == 0:
                        _log_warning(f"[Warning] {motion_loader_name} batch {idx}: no valid distances left, skipped.", file=file)
                        continue
                    dist_mat = dist_mat[np.ix_(valid_idx, valid_idx)]
                    motion_embeddings = motion_embeddings[valid_idx]
                # print(dist_mat.shape)
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                # print(argsmax.shape)

                cur_top_k = min(3, argsmax.shape[1])
                top_k_mat = calculate_top_k(argsmax, top_k=cur_top_k)
                if cur_top_k < 3:
                    pad = np.zeros((top_k_mat.shape[0], 3 - cur_top_k), dtype=top_k_mat.dtype)
                    top_k_mat = np.concatenate([top_k_mat, pad], axis=1)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings)

            if len(all_motion_embeddings) > 0:
                all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            else:
                all_motion_embeddings = np.empty((0, 0), dtype=np.float32)

            if all_size > 0:
                mm_dist = mm_dist_sum / all_size
                R_precision = top_k_count / all_size
            else:
                _log_warning(f"[Warning] {motion_loader_name}: no valid samples found for matching metrics.", file=file)
                mm_dist = np.nan
                R_precision = np.zeros(3, dtype=np.float64)
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch).cpu().numpy()
            motion_embeddings = _sanitize_embedding_rows(
                motion_embeddings, f'ground truth batch {idx} motion embeddings', file=file, min_rows=0
            )
            if motion_embeddings.shape[0] > 0:
                gt_motion_embeddings.append(motion_embeddings)
    if len(gt_motion_embeddings) == 0:
        _log_warning('[Warning] Ground-truth FID embeddings are empty after sanitization; reporting NaN FID.', file=file)
        for model_name in activation_dict.keys():
            fid = np.nan
            print(f'---> [{model_name}] FID: {fid:.4f}')
            print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
            eval_dict[model_name] = fid
        return eval_dict

    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_motion_embeddings = _sanitize_embedding_rows(gt_motion_embeddings, 'ground truth FID embeddings', file=file, min_rows=2)
    if gt_motion_embeddings.shape[0] < 2:
        _log_warning('[Warning] Ground-truth FID embeddings have fewer than 2 valid rows; reporting NaN FID.', file=file)
        for model_name in activation_dict.keys():
            fid = np.nan
            print(f'---> [{model_name}] FID: {fid:.4f}')
            print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
            eval_dict[model_name] = fid
        return eval_dict

    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings, emb_scale)
    if (not np.isfinite(gt_mu).all()) or (not np.isfinite(gt_cov).all()):
        _log_warning('[Warning] Ground-truth activation statistics contain NaN/Inf; reporting NaN FID.', file=file)
        for model_name in activation_dict.keys():
            fid = np.nan
            print(f'---> [{model_name}] FID: {fid:.4f}')
            print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
            eval_dict[model_name] = fid
        return eval_dict

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        motion_embeddings = _sanitize_embedding_rows(motion_embeddings, f'{model_name} FID embeddings', file=file, min_rows=2)
        if motion_embeddings.shape[0] < 2:
            _log_warning(f'[Warning] {model_name}: insufficient valid FID embeddings; reporting NaN.', file=file)
            fid = np.nan
        elif motion_embeddings.shape[1] != gt_motion_embeddings.shape[1]:
            _log_warning(
                f"[Warning] {model_name}: embedding dim {motion_embeddings.shape[1]} mismatches ground truth dim {gt_motion_embeddings.shape[1]}; reporting NaN.",
                file=file,
            )
            fid = np.nan
        else:
            mu, cov = calculate_activation_statistics(motion_embeddings, emb_scale)
            if (not np.isfinite(mu).all()) or (not np.isfinite(cov).all()):
                _log_warning(f'[Warning] {model_name}: activation statistics contain NaN/Inf; reporting NaN FID.', file=file)
                fid = np.nan
            else:
                try:
                    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
                except Exception as e:
                    _log_warning(f'[Warning] {model_name}: FID computation failed ({e}); reporting NaN.', file=file)
                    fid = np.nan
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        motion_embeddings = _sanitize_embedding_rows(motion_embeddings, f'{model_name} diversity embeddings', file=file, min_rows=2)
        if motion_embeddings.shape[0] < 2:
            _log_warning(f'[Warning] {model_name}: insufficient valid embeddings for diversity; reporting NaN.', file=file)
            diversity = np.nan
        else:
            diversity_times_eff = min(diversity_times, motion_embeddings.shape[0] - 1)
            if diversity_times_eff <= 0:
                _log_warning(f'[Warning] {model_name}: effective diversity_times <= 0; reporting NaN.', file=file)
                diversity = np.nan
            else:
                diversity = calculate_diversity(motion_embeddings, diversity_times_eff, emb_scale, divide_by)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                if len(batch) == 5:
                    batch[2] = batch[2][0]
                    batch[3] = batch[3][0]
                    batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch)
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times, emb_scale, divide_by)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            if replication > 0:
                opt.save_vis = False
            motion_loaders['ground truth'].dataset.normalize = True
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                print(f'Generating motions from {motion_loader_name}')
                # from ipdb import set_trace; set_trace()
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                if mm_motion_loader is not None:
                    mm_motion_loaders[motion_loader_name] = mm_motion_loader
            motion_loaders['ground truth'].dataset.normalize = False

            print(f'\n==================== Replication {replication} ====================')
            print(f'\n==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            if mm_motion_loaders:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!\n')
            print(f'!!! DONE !!!\n', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['MM Distance']:
                    all_metrics['MM Distance'][key] = [item]
                else:
                    all_metrics['MM Distance'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            if mm_motion_loaders:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

def evaluation_during_training(opt, net, test_loader, eval_wrapper_passed, epoch, file, trans=None):
    mm_num_samples = 0 #100
    mm_num_repeats = 30
    time_steps = 20
    cond_scale = 2
    topkr = 0.9

    global eval_wrapper, emb_scale, divide_by
    eval_wrapper = eval_wrapper_passed

    test_loader.dataset.normalize = True
    if opt.dataset_name == "interhuman":
        from models.evaluator.evaluator import get_motion_loader
        emb_scale = 6
        divide_by = 2
    elif opt.dataset_name == "interx":
        from models.evaluator.evaluator_interx import get_motion_loader
        emb_scale = 1
        divide_by = 1

    opt.gen_react = False
    gen_motion_loader, _ = get_motion_loader(
                            opt.test_batch_size,
                            net,
                            trans,
                            test_loader.dataset,
                            opt.device,
                            mm_num_samples,
                            mm_num_repeats,
                            None,
                            opt,
                            time_steps,
                            cond_scale,
                            topkr
                            )
    
    test_loader.dataset.normalize = False
    eval_motion_loaders = {'gt': test_loader,
                           'gen': gen_motion_loader}
    
    with open(file, 'a') as f:
        print(f'==================== Epoch {epoch} ====================')
        print(f'\n==================== Epoch {epoch} ====================', file=f, flush=True)

        mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_motion_loaders, f)

        fid_score_dict = evaluate_fid(test_loader, acti_dict, f)

    return fid_score_dict['gen'], mat_score_dict['gen'], R_precision_dict['gen'][0]
    


def load_vq_model(vq_opt, which_epoch):
    model_dir = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, "model")
    ckpt_path = pjoin(model_dir, which_epoch)
    if not os.path.isfile(ckpt_path):
        available = sorted(os.listdir(model_dir)) if os.path.isdir(model_dir) else []
        raise FileNotFoundError(f"Unified VQ checkpoint not found: {ckpt_path}. Available: {available}")

    vq_model, vq_epoch, ckpt_file = load_unified_vq_model(vq_opt, ckpt_path, device=opt.device)
    print(f"Loading Unified VQ Model {vq_opt.name} from {ckpt_file} Completed!, Epoch {vq_epoch}")
    return vq_model, vq_epoch

def load_trans_model(model_opt, which_model):
    # clip_version = 'ViT-B/32'
    # clip_version = 'ViT-L/14@336px'
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
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

if __name__ == '__main__':
    opt = arg_parse()
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    if opt.use_trans:
        trans_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
        main_opt = get_opt(trans_opt_path, opt.device)
        fixseed(main_opt.seed)

        vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, main_opt.vq_name, 'opt.txt')
        vq_opt = get_opt(vq_opt_path, opt.device)
        ensure_unified_vq_opt(vq_opt, getattr(main_opt, "vq_name", None))

        main_opt.num_tokens = vq_opt.nb_code
        main_opt.code_dim = vq_opt.code_dim
    else:
        opt.mm_num_samples = 0
        vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt')
        main_opt = get_opt(vq_opt_path, opt.device)
        ensure_unified_vq_opt(main_opt, main_opt.name)
    
    mm_num_samples = opt.mm_num_samples
    mm_num_repeats = opt.mm_num_repeats
    mm_num_times = opt.mm_num_times
    diversity_times = opt.diversity_times
    replication_times = opt.replication_times

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.vis_dir = pjoin(opt.save_root, 'animation')
    text_source_override = _resolve_text_source_override(opt)
    
    if opt.dataset_name == "interhuman":
        opt.npy_dir = pjoin(opt.vis_dir, 'keypoint_npy')
        opt.vis_dir = pjoin(opt.vis_dir, 'keypoint_mp4')
        os.makedirs(opt.npy_dir, exist_ok=True)
    elif opt.dataset_name == "interx":
        opt.vis_dir = pjoin(opt.vis_dir, 'smpl_npy')
    
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.vis_dir, exist_ok=True)
    react_name = "react_" if opt.gen_react else ""     

    if main_opt.dataset_name == "interhuman":
        main_opt.data_root = 'data/InterHuman'
        main_opt.joints_num = 22
        dim_pose = 12
        fps = 30
        opt.batch_size = 96
        main_opt.mode = "test"
        emb_scale = 6
        divide_by = 2
        
        from models.evaluator.evaluator import EvaluatorModelWrapper, get_dataset_motion_loader, get_motion_loader
        evalmodel_cfg = get_opt("checkpoints/eval_model/eval_model.yaml", opt.device, complete=False)
        eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, opt.device)

    elif main_opt.dataset_name == "interx":
        main_opt.data_root = 'data/Inter-X_Dataset'
        opt.data_root = main_opt.data_root
        if hasattr(main_opt, 'data_rep'):
            main_opt.data_rep = str(main_opt.data_rep).lower()
        elif 'vq_opt' in locals():
            main_opt.data_rep = str(getattr(vq_opt, 'data_rep', 'rot6d')).lower()
        else:
            main_opt.data_rep = 'rot6d'
        if not hasattr(main_opt, 'use_processed_loader') and 'vq_opt' in locals():
            main_opt.use_processed_loader = getattr(vq_opt, 'use_processed_loader', False)
        default_motion_dir = pjoin(main_opt.data_root, 'processed/motions')
        if getattr(main_opt, 'use_processed_loader', False):
            candidate_motion_dir = pjoin(main_opt.data_root, f'processed/motions_{main_opt.data_rep}')
            if os.path.isdir(candidate_motion_dir):
                main_opt.motion_dir = candidate_motion_dir
            else:
                print(f"[Warning] Processed motion dir {candidate_motion_dir} not found. Falling back to {default_motion_dir}.")
                main_opt.motion_dir = default_motion_dir
        else:
            main_opt.motion_dir = default_motion_dir
        apply_interx_text_config(main_opt, text_source=text_source_override, require_exists=True)
        main_opt.motion_rep = "smpl"
        main_opt.joints_num = 56 
        if 'vq_opt' in locals():
            dim_pose = getattr(vq_opt, 'dim_joint', 6)
        else:
            dim_pose = getattr(main_opt, 'dim_joint', 6)
        fps = 30
        opt.batch_size = 32
        main_opt.max_motion_length = 150
        main_opt.unit_length = 4
        emb_scale = 1
        divide_by = 1

        from models.evaluator.evaluator_interx import EvaluatorModelWrapper, get_dataset_motion_loader, get_motion_loader
        wrapper_opt = get_opt("checkpoints/interx/text_mot_match/model/opt.txt", opt.device, complete=False)
        wrapper_opt.data_rep = main_opt.data_rep
        wrapper_opt.max_text_len = main_opt.max_text_len
        wrapper_opt.text_mot_match_name = resolve_interx_eval_model_name(main_opt.text_source)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    else:
        raise KeyError('Dataset Does not Exists')
    
    
    data_cfg = main_opt
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, opt.batch_size)

    def make_callable(net, file, trans=None):
        return lambda: get_motion_loader(
                                        opt.batch_size,
                                        net,
                                        trans,
                                        gt_dataset,
                                        opt.device,
                                        mm_num_samples,
                                        mm_num_repeats,
                                        file,
                                        opt,
                                        time_step,
                                        cond_scale,
                                        topkr
                                        )
    
    if opt.use_trans:
        for cond_scale in opt.cond_scales:
            for time_step in opt.time_steps:
                for topkr in opt.topkr:
                    eval_motion_loaders = {}
                    for file in os.listdir(opt.model_dir):
                        if opt.which_epoch != "all" and opt.which_epoch not in file:
                            continue
                        
                        print(f"\n\nLoading model epoch: {file}")
                        trans = load_trans_model(main_opt, file)
                        vq_epoch_file = "finest.tar" if main_opt.dataset_name == "interx" else "best_fid.tar"
                        net, ep = load_vq_model(vq_opt, vq_epoch_file)
                        
                        file = react_name + file
                        eval_motion_loaders[file] = make_callable(net, file, trans)
                    
                    which_epoch = opt.which_epoch
                    
                    log_file_name = f'evaluation_{which_epoch}_ts{time_step}_cs{cond_scale}_topkr{topkr}.log'
                    log_file_name = react_name + log_file_name
                    log_file = pjoin(opt.eval_dir, log_file_name)
                    evaluation(log_file)
    else:
        eval_motion_loaders = {}
        for file in os.listdir(opt.model_dir):
            if opt.which_epoch != "all" and opt.which_epoch not in file:
                continue
            cond_scale, time_step, topkr = None, None, None
            
            print(f"\n\nLoading model epoch: {file}")
            net, ep = load_vq_model(main_opt, file)
            eval_motion_loaders[file] = make_callable(net, file)
            
        log_file = pjoin(opt.eval_dir, f'evaluation_{opt.which_epoch}.log')
        evaluation(log_file)
