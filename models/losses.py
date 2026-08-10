import torch
from utils.paramUtil import t2m_kinematic_chain as kinematic_chain
from data.quaternion import *
from data.utils import *
from data.interx_utils import *
import data.rotation_conversions as geometry


NONCANONICAL_JOINTS = 55
NONCANONICAL_POS_DIM = NONCANONICAL_JOINTS * 3
NONCANONICAL_VEL_DIM = NONCANONICAL_JOINTS * 3
NONCANONICAL_ROT_DIM = NONCANONICAL_JOINTS * 6
NONCANONICAL_FC_DIM = 4
NONCANONICAL_PERSON_DIM = NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM + NONCANONICAL_ROT_DIM + NONCANONICAL_FC_DIM


class Geometric_Losses:
    def __init__(self, recons_loss, joints_num, dataset_name, device, data_rep='rot6d', fast_mode=False):
        
        if recons_loss == 'l1':
            self.l1_criterion = torch.nn.L1Loss()
        elif recons_loss == 'l1_smooth':
            self.l1_criterion = torch.nn.SmoothL1Loss()
        
        self.joints_num = joints_num
        self.fids = [*fid_l, *fid_r]
        self.dataset_name = dataset_name
        self.fast_mode = fast_mode
        self.data_rep = str(data_rep).lower() if self.dataset_name == 'interx' else None
        if self.dataset_name == 'interhuman':
            self.normalizer = MotionNormalizerTorch(device)
        elif self.dataset_name == 'interx':
            self.normalizer = InterxNormalizerTorch()
            self.kinematics = InterxKinematicsFlexible(data_rep=data_rep)

    def _split_noncanonical_flat(self, motions):
        """
        Parse one-person noncanonical flat features:
            [B, T, 664] -> pos[B,T,55,3], vel[B,T,55,3], rot[B,T,55,6], fc[B,T,4]
        """
        if motions.ndim != 3 or motions.shape[-1] != NONCANONICAL_PERSON_DIM:
            raise ValueError(
                f"Expected noncanonical motion shape [B, T, {NONCANONICAL_PERSON_DIM}], got {motions.shape}"
            )
        pos = motions[..., :NONCANONICAL_POS_DIM].reshape(motions.shape[0], motions.shape[1], NONCANONICAL_JOINTS, 3)
        vel = motions[..., NONCANONICAL_POS_DIM:NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM].reshape(
            motions.shape[0], motions.shape[1], NONCANONICAL_JOINTS, 3
        )
        rot = motions[..., NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM:
                      NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM + NONCANONICAL_ROT_DIM].reshape(
            motions.shape[0], motions.shape[1], NONCANONICAL_JOINTS, 6
        )
        fc = motions[..., -NONCANONICAL_FC_DIM:]
        return pos, vel, rot, fc

    def calc_foot_contact(self, motion, pred_motion):
            if self.dataset_name == 'interhuman':
                B, T, _ = motion.shape
                motion = motion[..., :self.joints_num * 3]
                motion = motion.reshape(B, T, self.joints_num, 3)
                
                pred_motion = pred_motion[..., :self.joints_num * 3]
                pred_motion = pred_motion.reshape(B, T, self.joints_num, 3)
            
            
            feet_vel = motion[:, 1:, self.fids, :] - motion[:, :-1, self.fids,:]
            pred_feet_vel = pred_motion[:, 1:, self.fids, :] - pred_motion[:, :-1, self.fids,:]
            feet_h = motion[:, :-1, self.fids, 1]
            pred_feet_h = pred_motion[:, :-1, self.fids, 1]
            # contact = target[:,:-1,:,-8:-4] # [b,t,p,4]

            ## Calculate contacts
            thres = 0.001
            velfactor, heightfactor = torch.Tensor([thres, thres, thres, thres]).to(feet_vel.device), torch.Tensor(
                [0.12, 0.05, 0.12, 0.05]).to(feet_vel.device)

            feet_x = (feet_vel[..., 0]) ** 2
            feet_y = (feet_vel[..., 1]) ** 2
            feet_z = (feet_vel[..., 2]) ** 2
            contact = ((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)
            valid_contact = contact & torch.isfinite(pred_feet_vel).all(dim=-1)
            
            if valid_contact.any():
                fc_loss = self.l1_criterion(pred_feet_vel[valid_contact], torch.zeros_like(pred_feet_vel)[valid_contact])
            else:
                fc_loss = torch.tensor(0.0, device=motion.device, dtype=motion.dtype)
                if contact.any():
                    invalid_count = int((contact & ~torch.isfinite(pred_feet_vel).all(dim=-1)).sum().item())
                    print(f"FC skipped due to non-finite predicted foot velocity on {invalid_count} contact frames")
            if torch.isnan(fc_loss):
                fc_loss = torch.tensor(0.0, device=motion.device, dtype=motion.dtype)
                if contact.sum() != 0:
                    print("FC nan but contact not 0")
            return fc_loss
                                 
    def calc_bone_lengths(self, motion):
        if self.dataset_name == 'interhuman':
            motion_pos = motion[..., :self.joints_num*3]
            motion_pos = motion_pos.reshape(motion_pos.shape[0], motion_pos.shape[1], self.joints_num, 3)
        elif self.dataset_name == 'interx':
            motion_pos = motion
        bones = []
        for chain in kinematic_chain:
            for i, joint in enumerate(chain[:-1]):
                bone = (motion_pos[..., chain[i], :] - motion_pos[..., chain[i + 1], :]).norm(dim=-1, keepdim=True)  # [B,T,P,1]
                bones.append(bone)

        return torch.cat(bones, dim=-1)

    def _rot_to_matrix(self, rot):
        """
        Convert rotational features to 3x3 matrices for geodesic loss.
        Conversion branch is chosen by `self.data_rep`.
        """
        if self.data_rep in ('rot6d', 'canonical', 'noncanonical'):
            if rot.shape[-1] != 6:
                raise ValueError(f"{self.data_rep} expects last dim 6, got {rot.shape}")
            return cont6d_to_matrix(rot)

        if self.data_rep == 'axis':
            if rot.shape[-1] != 3:
                raise ValueError(f"axis expects last dim 3, got {rot.shape}")
            return geometry.axis_angle_to_matrix(rot)

        if self.data_rep == 'quaternion':
            if rot.shape[-1] != 4:
                raise ValueError(f"quaternion expects last dim 4, got {rot.shape}")
            quat = rot / torch.clamp(rot.norm(dim=-1, keepdim=True), min=1e-8)
            return geometry.quaternion_to_matrix(quat)

        if self.data_rep == 'matrix':
            if rot.shape[-1] == 9:
                return rot.view(*rot.shape[:-1], 3, 3)
            if rot.shape[-2:] == (3, 3):
                return rot
            raise ValueError(f"matrix expects (..., 9) or (..., 3, 3), got {rot.shape}")

        raise ValueError(f"Unsupported data_rep for geodesic loss: {self.data_rep}")
    
    def calc_loss_geo(self, pred_rot, gt_rot, eps=1e-7):
        if self.dataset_name == "interhuman":
            pred_rot = pred_rot.reshape(pred_rot.shape[0], pred_rot.shape[1], -1, 6)
            gt_rot = gt_rot.reshape(gt_rot.shape[0], gt_rot.shape[1], -1, 6)
            pred_m = cont6d_to_matrix(pred_rot).reshape(-1,3,3)
            gt_m = cont6d_to_matrix(gt_rot).reshape(-1,3,3)
        else:
            if self.data_rep == 'joint':
                return torch.zeros(1, device=pred_rot.device, dtype=pred_rot.dtype)
            pred_m = self._rot_to_matrix(pred_rot).reshape(-1, 3, 3)
            gt_m = self._rot_to_matrix(gt_rot).reshape(-1, 3, 3)

        m = torch.bmm(gt_m, pred_m.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+eps, 1-eps))

        return torch.mean(theta)
    
    def forward(self, motions, pred_motion):
        if self.dataset_name == 'interhuman':
            loss_rec = self.l1_criterion(pred_motion[..., :-4], motions[..., :-4])
            
            loss_explicit = self.l1_criterion(pred_motion[:, :, :self.joints_num*3],
                                            motions[:, :, :self.joints_num*3])
            
            loss_vel = self.l1_criterion(pred_motion[:, 1:, :self.joints_num*3] - pred_motion[:, :-1, :self.joints_num*3],
                                        motions[:, 1:, :self.joints_num*3] - motions[:, :-1, :self.joints_num*3])
            
            loss_bn = self.l1_criterion(self.calc_bone_lengths(pred_motion), self.calc_bone_lengths(motions))

            loss_geo = self.calc_loss_geo(pred_motion[..., self.joints_num*6: self.joints_num*6 + (self.joints_num-1)*6],
                                        motions[..., self.joints_num*6: self.joints_num*6 + (self.joints_num-1)*6])
            
            loss_fc = self.calc_foot_contact(self.normalizer.backward(motions), self.normalizer.backward(pred_motion))
            
            return loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, None, None
        elif self.dataset_name == 'interx':
            if self.fast_mode:
                zero = torch.zeros(1, device=motions.device, dtype=motions.dtype)
                loss_rec = self.l1_criterion(pred_motion, motions)
                loss_explicit = loss_vel = loss_bn = loss_geo = loss_fc = zero
                return loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, None, None

            loss_rec = self.l1_criterion(pred_motion, motions)
            if self.data_rep == 'noncanonical' and motions.ndim == 3:
                motions_pos, _, motions_rot, _ = self._split_noncanonical_flat(motions)
                pred_motions_pos, _, pred_motions_rot, _ = self._split_noncanonical_flat(pred_motion)

                loss_explicit = self.l1_criterion(pred_motions_pos, motions_pos)
                loss_vel = self.l1_criterion(
                    pred_motions_pos[:, 1:, :, :] - pred_motions_pos[:, :-1, :, :],
                    motions_pos[:, 1:, :, :] - motions_pos[:, :-1, :, :]
                )
                loss_bn = self.l1_criterion(
                    self.calc_bone_lengths(pred_motions_pos[:, :, :22, :]),
                    self.calc_bone_lengths(motions_pos[:, :, :22, :])
                )
                loss_geo = self.calc_loss_geo(pred_motions_rot, motions_rot)
                loss_fc = self.calc_foot_contact(motions_pos, pred_motions_pos)
                return loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, motions_pos, pred_motions_pos

            pred_motions_pos = self.kinematics.forward(pred_motion)
            motions_pos = self.kinematics.forward(motions)

            loss_explicit = self.l1_criterion(pred_motions_pos, motions_pos)

            loss_vel = self.l1_criterion(pred_motions_pos[:,1:,:,:] - pred_motions_pos[:,:-1,:,:],
                                            motions_pos[:,1:,:,:] - motions_pos[:,:-1,:,:])
            
            loss_bn = self.l1_criterion(self.calc_bone_lengths(pred_motions_pos[:,:,:22,:]), self.calc_bone_lengths(motions_pos[:,:,:22,:]))

            loss_geo = self.calc_loss_geo(pred_motion[:,:,:-1,:], motions[:,:,:-1,:])

            loss_fc = self.calc_foot_contact(motions_pos, pred_motions_pos)


            return loss_rec, loss_explicit, loss_vel, loss_bn, loss_geo, loss_fc, motions_pos, pred_motions_pos
    

class Inter_Losses:
    def __init__(self, recons_loss, joints_num, dataset_name, device, data_rep='rot6d', fast_mode=False):
        self.dataset_name = dataset_name
        if recons_loss == 'l1':
            self.l1_criterion = torch.nn.L1Loss('none')
        elif recons_loss == 'l1_smooth':
            self.l1_criterion = torch.nn.SmoothL1Loss(reduction='none')
        
        self.joints_num = joints_num
        self.fast_mode = fast_mode
        self.data_rep = str(data_rep).lower() if self.dataset_name == 'interx' else None
        if self.dataset_name == 'interhuman':
            self.normalizer = MotionNormalizerTorch(device)
        elif self.dataset_name == 'interx':
            self.normalizer = InterxNormalizerTorch()
            self.kinematics = InterxKinematicsFlexible(data_rep=data_rep)

    def _split_noncanonical_flat(self, motions):
        if motions.ndim != 3 or motions.shape[-1] != NONCANONICAL_PERSON_DIM:
            raise ValueError(
                f"Expected noncanonical motion shape [B, T, {NONCANONICAL_PERSON_DIM}], got {motions.shape}"
            )
        pos = motions[..., :NONCANONICAL_POS_DIM].reshape(motions.shape[0], motions.shape[1], NONCANONICAL_JOINTS, 3)
        vel = motions[..., NONCANONICAL_POS_DIM:NONCANONICAL_POS_DIM + NONCANONICAL_VEL_DIM].reshape(
            motions.shape[0], motions.shape[1], NONCANONICAL_JOINTS, 3
        )
        return pos, vel
    
    def calc_dm_loss(self, motion_joints, pred_motion_joints, thresh_pred=1, thresh_tgt=0.1):

        pred_motion_joints1 = pred_motion_joints[..., 0:1, :, :].reshape(-1, self.joints_num, 3)
        pred_motion_joints2 = pred_motion_joints[..., 1:2, :, :].reshape(-1, self.joints_num, 3)
        motion_joints1 = motion_joints[..., 0:1, :, :].reshape(-1, self.joints_num, 3)
        motion_joints2 = motion_joints[..., 1:2, :, :].reshape(-1, self.joints_num, 3)
        
        pred_distance_matrix = torch.cdist(pred_motion_joints1.contiguous(), pred_motion_joints2)
        tgt_distance_matrix = torch.cdist(motion_joints1.contiguous(), motion_joints2)
        
        pred_distance_matrix = pred_distance_matrix.reshape(pred_distance_matrix.shape[0], -1).reshape(self.B, self.T, self.joints_num*self.joints_num) # B*T, njoints=22, 22 -> B, T, 484
        tgt_distance_matrix = tgt_distance_matrix.reshape(pred_distance_matrix.shape[0], -1).reshape(self.B, self.T, self.joints_num*self.joints_num)
        
        dm_mask = (pred_distance_matrix < thresh_pred).float()
        dm_tgt_mask = (tgt_distance_matrix < thresh_tgt).float()
        
        dm_loss = (self.l1_criterion(pred_distance_matrix, tgt_distance_matrix) * dm_mask).sum() / (dm_mask.sum() + 1.e-7)
        dm_tgt_loss = (self.l1_criterion(pred_distance_matrix, torch.zeros_like(tgt_distance_matrix)) * dm_tgt_mask).sum()/ (dm_tgt_mask.sum() + 1.e-7)
        
        return dm_loss + dm_tgt_loss
    
    def calc_ro_loss(self, motion_joints, pred_motion_joints):

        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across = pred_motion_joints[..., r_hip, :] - pred_motion_joints[..., l_hip, :]
        across = across / across.norm(dim=-1, keepdim=True)
        across_gt = motion_joints[..., r_hip, :] - motion_joints[..., l_hip, :]
        across_gt = across_gt / across_gt.norm(dim=-1, keepdim=True)

        y_axis = torch.zeros_like(across)
        y_axis[..., 1] = 1

        forward = torch.cross(y_axis, across, axis=-1)
        forward = forward / forward.norm(dim=-1, keepdim=True)
        forward_gt = torch.cross(y_axis, across_gt, axis=-1)
        forward_gt = forward_gt / forward_gt.norm(dim=-1, keepdim=True)

        pred_relative_rot = qbetween(forward[..., 0, :], forward[..., 1, :])
        tgt_relative_rot = qbetween(forward_gt[..., 0, :], forward_gt[..., 1, :])

        ro_loss = self.l1_criterion(pred_relative_rot[..., [0, 2]],
                                    tgt_relative_rot[..., [0, 2]]).mean()

        return ro_loss
    
    def forward(self, motion1, motion2, pred_motion1, pred_motion2):
        B, T = motion1.shape[:2]
        self.B = B
        self.T = T
        
        if self.dataset_name == 'interhuman':
            motions = torch.cat([motion1.unsqueeze(-2), motion2.unsqueeze(-2)], dim=-2)
            motions = self.normalizer.backward(motions)
            
            pred_motion = torch.cat([pred_motion1.unsqueeze(-2), pred_motion2.unsqueeze(-2)], dim=-2)
            pred_motion = self.normalizer.backward(pred_motion)
            
            pred_motion_joints = pred_motion[..., :self.joints_num * 3].reshape(B, T, -1, self.joints_num, 3)
            motion_joints = motions[..., :self.joints_num * 3].reshape(B, T, -1, self.joints_num, 3)
        elif self.dataset_name == 'interx':
            if self.fast_mode:
                zero = torch.zeros(1, device=motion1.device, dtype=motion1.dtype)
                return zero, zero
            if self.data_rep == 'noncanonical' and motion1.ndim == 3:
                motion1_pos, _ = self._split_noncanonical_flat(motion1)
                motion2_pos, _ = self._split_noncanonical_flat(motion2)
                pred_motion1_pos, _ = self._split_noncanonical_flat(pred_motion1)
                pred_motion2_pos, _ = self._split_noncanonical_flat(pred_motion2)
                self.joints_num = motion1_pos.shape[-2]
                motion_joints = torch.cat([motion1_pos.unsqueeze(2), motion2_pos.unsqueeze(2)], dim=2)
                pred_motion_joints = torch.cat([pred_motion1_pos.unsqueeze(2), pred_motion2_pos.unsqueeze(2)], dim=2)
            else:
                motion_joints = torch.cat([motion1.unsqueeze(2), motion2.unsqueeze(2)], dim=2)
                pred_motion_joints = torch.cat([pred_motion1.unsqueeze(2), pred_motion2.unsqueeze(2)], dim=2)
        
        ro_loss = self.calc_ro_loss(motion_joints, pred_motion_joints)
        dm_loss = self.calc_dm_loss(motion_joints, pred_motion_joints)
        
        return dm_loss, ro_loss
