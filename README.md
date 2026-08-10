# OpenHHI

**OpenHHI** is a text-driven human-human interaction generation framework. This release targets the Inter-X++ dataset and includes the unified VQ tokenizer, Masked Transformer, training and inference pipelines, quantitative evaluation tools, rot6d motion data, and the minimum set of pretrained checkpoints required to run the framework.

## Repository Structure

    OpenHHI/
    ├── checkpoints/                  # Models required for text-to-interaction generation
    ├── data/
    │   ├── Inter-X_Dataset/
    │   │   ├── processed/
    │   │   │   ├── motions_rot6d/   # Train/validation/test HDF5 files
    │   │   │   ├── texts_processed/
    │   │   │   └── glove/
    │   │   └── splits/
    │   ├── body_model/smplx/SMPLX_NEUTRAL.npz
    │   └── stats/
    ├── models/
    ├── options/
    └── utils/

## Environment Setup

    cd /path/to/OpenHHI
    conda env create -f environment.yml
    conda activate openhhi

The environment uses Python 3.9, PyTorch 1.13.1, and CUDA 11.7. Run all commands from the repository root because data and checkpoint paths are relative to it.

## Download Models and Data

Download the following three archives from [Google Drive](https://drive.google.com/drive/folders/1hH1UyCKOX77QXsh183GkfUkf94y7Oz4g?usp=sharing):

| Archive | Contents | Destination |
|---|---|---|
| `vq_rot6d.zip` | Pretrained vanilla VQ checkpoint used to initialize Stage 1 training | Extract into `checkpoints/interx/` |
| `checkpoints.zip` | Trained OpenHHI generation models and checkpoints required for inference and evaluation | Extract into the repository root |
| `data.zip` | Processed Inter-X++ data and supporting data files used for training and evaluation | Extract into the repository root |

After extraction, verify that the repository contains these paths:

    checkpoints/interx/vq_rot6d/model/finest.tar
    checkpoints/interx/vq_rot6d_uni/model/finest.tar
    checkpoints/interx/trans_rot6d_uni/model/best_fid.tar
    data/Inter-X_Dataset/processed/motions_rot6d/

## Text-Driven Generation

Edit `prompts.txt` and place one English interaction description on each line. Then run:

    python infer_uni.py --dataset_name interx --name trans_rot6d_uni --which_epoch best_fid --gpu_id 0

To specify the sampling parameters explicitly:

    python infer_uni.py --dataset_name interx --name trans_rot6d_uni --which_epoch best_fid --time_steps 20 --cond_scales 2 --topkr 0.9 --gpu_id 0

Generated motions are written to `checkpoints/interx/trans_rot6d_uni/animation_infer/smpl_npy/`. Each `.npy` file contains a pickled dictionary with `root_trans`, `root_vel`, `root_orient`, and `body_pose` for both people.

To randomly sample 200 captions from the Inter-X++ test split and generate interactions:

    python infer_uni_random.py --dataset_name interx --name trans_rot6d_uni --which_epoch best_fid --gpu_id 0

Outputs are written to `outputs/openhhi_random_infer/trans_rot6d_uni/runs/<timestamp>/`.

## Quantitative Evaluation

    python eval_uni.py --dataset_name interx --name trans_rot6d_uni --which_epoch best_fid --gpu_id 0

This command evaluates Matching Score, R-Precision, FID, Diversity, and Multimodality. Evaluation uses 20 replications by default.

Evaluation logs are written to `checkpoints/interx/trans_rot6d_uni/eval/`.

## Training

### Stage 1: Unified VQ Tokenizer

Extract `vq_rot6d.zip` as described in [Download Models and Data](#download-models-and-data), and verify that the pretrained vanilla VQ checkpoint is located at:

    checkpoints/interx/vq_rot6d/model/finest.tar

Then train the unified VQ tokenizer:

    python train_vq_uni.py --dataset_name interx --name vq_rot6d_uni_new --data_rep rot6d --use_processed_loader --init_vq_ckpt checkpoints/interx/vq_rot6d/model/finest.tar --batch_size 128 --gpu_id 0

The training script uses this checkpoint path by default. After loading the checkpoint, it freezes the convolutional encoder and trains the remaining unified VQ components.

### Stage 2: Masked Transformer

Train the Masked Transformer with the released unified VQ tokenizer:

    python train_transformer_uni.py --dataset_name interx --name trans_rot6d_uni_new --vq_name vq_rot6d_uni --data_rep rot6d --use_processed_loader --batch_size 128 --gpu_id 0

Transformer training evaluates the validation split by default. Pass `--no_eval` to disable validation-time evaluation.

The released `vq_rot6d_uni` was initialized from the checkpoint in `vq_rot6d.zip`, with its convolutional encoder frozen during training. The released `trans_rot6d_uni` was trained from a random initialization; `--load_pretrained_trans` is disabled by default.


## Citation
If you find the Inter-X/Inter-X++ dataset is useful for your research, please cite us:

```
@inproceedings{xu2024inter,
  title={Inter-x: Towards versatile human-human interaction analysis},
  author={Xu, Liang and Lv, Xintao and Yan, Yichao and Jin, Xin and Wu, Shuwen and Xu, Congsheng and Liu, Yifan and Zhou, Yizhou and Rao, Fengyun and Sheng, Xingdong and others},
  booktitle={CVPR},
  pages={22260--22271},
  year={2024}
}
```