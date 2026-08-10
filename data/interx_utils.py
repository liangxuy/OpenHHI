import numpy as np
import torch
from data.body_model.body_model import BodyModel
import data.rotation_conversions as geometry


def _safe_normalize_quaternion(quat, eps=1e-8):
    """
    Normalize quaternions robustly.
    Invalid rows (NaN/Inf or near-zero norm) fall back to identity quaternion.
    """
    quat = torch.as_tensor(quat)
    norm = quat.norm(dim=-1, keepdim=True)
    finite_mask = torch.isfinite(quat).all(dim=-1, keepdim=True)
    valid_mask = finite_mask & (norm > eps)

    safe_norm = torch.where(valid_mask, norm, torch.ones_like(norm))
    quat = quat / safe_norm

    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    quat = torch.where(valid_mask.expand_as(quat), quat, identity)
    return quat


class InterxKinematics():
    def __init__(self):
        self.bm = BodyModel(bm_fname="data/body_model/smplx/SMPLX_NEUTRAL.npz", num_betas=10)
        self.bm.eval()
    
    def rot6d_to_axisangle(self, motionrot6d):
        # Root Translation and Velocity       
        root = motionrot6d[:, :, -1, :]
        root_trans = root[:, :, :3]
        root_vel = root[:, :, 3:]

        # Whole body pose
        pose = motionrot6d[:, :, :-1, :]
        pose = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(pose))

        # Root Orientation
        root_orient = pose[:, :, 0, :]
        body_pose = pose[:, :, 1:, :]

        return root_trans, root_vel, root_orient, body_pose
    
    def forward(self, motions):
        """
        Args:
            motions: torch.Tensor of shape (B, T, 56, dim) 
            T=64 for vqvae
            dim=6 for 6d rots
        """
        B, T, J, dim = motions.shape
        self.bm.to(motions.device)
        
        root_trans, root_vel, root_orient, body_pose = self.rot6d_to_axisangle(motions)
        motions_pos = []
        #torch.zeros(B, T, J-1, 3, requires_grad=True).to(motions.device)
        for b in range(B):
            bm_output = self.bm(root_orient = root_orient[b],
                                    pose_body= body_pose[b, :, 0:21, :].reshape(T, -1),
                                    pose_hand = body_pose[b, :, 24:, :].reshape(T, -1))
            motions_pos.append(bm_output.Jtr + root_trans[b].unsqueeze(-2))
            # motions_pos[b] = bm_output.Jtr + root_trans[b].unsqueeze(-2)
        motions_pos = torch.stack(motions_pos, dim=0)
        return motions_pos

class InterxKinematicsFlexible():
    """
    A kinematics helper that accepts multiple motion representations and
    converts them to joint positions via SMPL-X.

    Supported `data_rep` values:
        rot6d, canonical, noncanonical, axis, quaternion, matrix, joint
    """
    SUPPORTED_REPRESENTATIONS = ('rot6d', 'canonical', 'noncanonical', 'axis', 'quaternion', 'joint', 'matrix')

    def __init__(self, data_rep='rot6d'):
        self.data_rep = 'rot6d' if data_rep is None else str(data_rep).lower()
        if self.data_rep not in self.SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"Unsupported data representation {data_rep}. Choose from {self.SUPPORTED_REPRESENTATIONS}")
        self.bm = BodyModel(bm_fname="data/body_model/smplx/SMPLX_NEUTRAL.npz", num_betas=10)
        self.bm.eval()

    def _split_motion(self, motions):
        """
        Splits motion into rotation representation and root translation/velocity.
        Assumes last joint stores root translation/velocity.
        """
        rot_block = motions[:, :, :-1, :]
        root_block = motions[:, :, -1, :]

        if root_block.shape[-1] >= 3:
            root_trans = root_block[..., :3]
        else:
            root_trans = torch.zeros((*root_block.shape[:-1], 3), device=motions.device, dtype=motions.dtype)

        if root_block.shape[-1] >= 6:
            root_vel = root_block[..., 3:6]
        else:
            root_vel = torch.zeros_like(root_trans)

        return rot_block, root_trans, root_vel

    def _rot_to_axisangle(self, rot_block):
        if self.data_rep in ('rot6d', 'canonical', 'noncanonical'):
            if rot_block.shape[-1] != 6:
                raise ValueError(f"{self.data_rep} expects last dim 6, got {rot_block.shape}")
            return geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(rot_block))

        if self.data_rep == 'axis':
            if rot_block.shape[-1] != 3:
                raise ValueError(f"axis expects last dim 3, got {rot_block.shape}")
            return rot_block

        if self.data_rep == 'quaternion':
            if rot_block.shape[-1] != 4:
                raise ValueError(f"quaternion expects last dim 4, got {rot_block.shape}")
            quat = _safe_normalize_quaternion(rot_block)
            return geometry.quaternion_to_axis_angle(quat)

        if self.data_rep == 'matrix':
            if rot_block.shape[-1] == 9:
                rot_mat = rot_block.view(*rot_block.shape[:-1], 3, 3)
                return geometry.matrix_to_axis_angle(rot_mat)
            if rot_block.shape[-2:] == (3, 3):
                return geometry.matrix_to_axis_angle(rot_block)
            raise ValueError(f"matrix expects (..., 9) or (..., 3, 3), got {rot_block.shape}")

        raise ValueError(f"Unsupported data_rep for axis-angle conversion: {self.data_rep}")

    def forward(self, motions):
        """
        Args:
            motions: torch.Tensor (B, T, J, D) with varying D depending on data_rep.
        Returns:
            motions_pos: torch.Tensor of joint positions, shaped like SMPL-X Jtr.
        """
        motions = torch.as_tensor(motions)
        B, T = motions.shape[:2]
        self.bm.to(motions.device)

        if self.data_rep == 'joint':
            # Assume joint positions are provided directly.
            positions = motions[:, :, :-1, :]
            if positions.shape[-1] > 3:
                positions = positions[..., :3]
            root_trans = motions[:, :, -1, :3]
            positions = positions + root_trans.unsqueeze(-2)
            return positions

        rot_block, root_trans, _ = self._split_motion(motions)
        axis_angles = self._rot_to_axisangle(rot_block)

        # Flatten the batch/time dims so the body model runs once per call.
        root_orient = axis_angles[:, :, 0, :].reshape(B * T, -1)
        body_pose = axis_angles[:, :, 1:, :]
        pose_body = body_pose[:, :, 0:21, :].reshape(B * T, -1)
        pose_hand = body_pose[:, :, 24:, :].reshape(B * T, -1)

        # from ipdb import set_trace; set_trace()
        bm_output = self.bm(root_orient=root_orient,
                            pose_body=pose_body,
                            pose_hand=pose_hand)

        joints = bm_output.Jtr + root_trans.reshape(B * T, 1, -1)
        motions_pos = joints.reshape(B, T, *joints.shape[1:])
        return motions_pos

class InterxNormalizerTorch():
    def __init__(self):
        mean = np.load("data/stats/interx_mean.npy")
        std = np.load("data/stats/interx_std.npy")

        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()


    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x

class InterxRot6dEvaluator():
    """
    Utility to unify evaluation by converting multiple motion representations
    to rot6d (and keeping root translation/velocity untouched).
    Supported `data_rep`: rot6d, canonical, noncanonical, axis, quaternion, matrix.
    """
    SUPPORTED_REPRESENTATIONS = ('rot6d', 'canonical', 'noncanonical', 'axis', 'quaternion', 'matrix')

    def __init__(self, data_rep='rot6d'):
        self.data_rep = 'rot6d' if data_rep is None else str(data_rep).lower()
        if self.data_rep not in self.SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"Unsupported data representation {data_rep}. Choose from {self.SUPPORTED_REPRESENTATIONS}")

    def _split_motion(self, motions):
        rot_block = motions[:, :, :-1, :]
        root_block = motions[:, :, -1, :]
        return rot_block, root_block

    def _to_rot6d(self, rot_block):
        if self.data_rep in ('rot6d', 'canonical', 'noncanonical'):
            if rot_block.shape[-1] != 6:
                raise ValueError(f"{self.data_rep} expects last dim 6, got {rot_block.shape}")
            return rot_block

        if self.data_rep == 'axis':
            if rot_block.shape[-1] != 3:
                raise ValueError(f"axis expects last dim 3, got {rot_block.shape}")
            return geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(rot_block))

        if self.data_rep == 'quaternion':
            if rot_block.shape[-1] != 4:
                raise ValueError(f"quaternion expects last dim 4, got {rot_block.shape}")
            quat = _safe_normalize_quaternion(rot_block)
            return geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(quat))

        if self.data_rep == 'matrix':
            if rot_block.shape[-1] == 9:
                rot_mat = rot_block.view(*rot_block.shape[:-1], 3, 3)
                return geometry.matrix_to_rotation_6d(rot_mat)
            if rot_block.shape[-2:] == (3, 3):
                return geometry.matrix_to_rotation_6d(rot_block)
            raise ValueError(f"matrix expects (..., 9) or (..., 3, 3), got {rot_block.shape}")

        raise ValueError(f"Unsupported data_rep for rot6d conversion: {self.data_rep}")

    def _normalize_root_block(self, root_block, target_dim=6):
        """
        Keep root translation and (optional) velocity in a fixed-width tensor.
        For axis/quaternion inputs (root_dim=3/4), velocity channels are zero-padded.
        """
        root_dim = root_block.shape[-1]
        if root_dim >= target_dim:
            return root_block[..., :target_dim]
        pad = torch.zeros(*root_block.shape[:-1], target_dim - root_dim,
                          device=root_block.device, dtype=root_block.dtype)
        return torch.cat([root_block, pad], dim=-1)

    def forward(self, motions):
        """
        Args:
            motions: torch.Tensor of shape (B, T, J, D).
        Returns:
            motions_rot6d: torch.Tensor with rotations in rot6d and root block
                           preserved (if present).
        """
        motions = torch.as_tensor(motions)
        rot_block, root_block = self._split_motion(motions)
        rot6d = self._to_rot6d(rot_block)
        root_block = torch.as_tensor(root_block)
        root_block = self._normalize_root_block(root_block, target_dim=rot6d.shape[-1])
        motions_rot6d = torch.cat([rot6d, root_block.unsqueeze(2)], dim=2)

        return motions_rot6d



        
