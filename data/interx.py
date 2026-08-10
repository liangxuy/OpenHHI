import codecs as cs
import h5py
import numpy as np
import random
import torch
from os.path import join as pjoin
from torch.utils import data
from tqdm import tqdm
import data.rotation_conversions as geometry
from data.body_model.body_model import BodyModel
from data.interx_utils import InterxNormalizerTorch
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def _load_split_ids(split_file):
    with cs.open(split_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def _load_text_data(text_dir, name):
    text_data = []
    with cs.open(pjoin(text_dir, name + '.txt')) as f:
        for line in f.readlines():
            line_split = line.strip().split('#')
            if len(line_split) < 4:
                continue
            caption = line_split[0]
            tokens = line_split[1].split(' ')
            f_tag = float(line_split[2])
            to_tag = float(line_split[3])
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            if f_tag == 0.0 and to_tag == 0.0:
                text_data.append({'caption': caption, 'tokens': tokens})
    return text_data


NONCANONICAL_JOINTS = 55
NONCANONICAL_PERSON_FLAT_DIM = NONCANONICAL_JOINTS * 3 * 2 + NONCANONICAL_JOINTS * 6 + 4
NONCANONICAL_TOTAL_FLAT_DIM = NONCANONICAL_PERSON_FLAT_DIM * 2
INTERX_MATRIX_SINGLE_DIM = 9
INTERX_MATRIX_TOTAL_DIM = INTERX_MATRIX_SINGLE_DIM * 2
INTERX_ROOT_BLOCK_DIM = 6


def _is_legacy_interx_matrix_layout(motion):
    """
    Detect the legacy Inter-X matrix packing bug.

    Expected two-person matrix layout:
        joint[0:55] -> [p1_rot9 | p2_rot9]
        joint[55]   -> [p1_root6, 0, 0, 0, p2_root6, 0, 0, 0]

    Legacy buggy layout found on disk:
        joint[55]   -> [p1_root6, p2_root6, 0, 0, 0, 0, 0, 0]
    """
    if motion.ndim != 3 or motion.shape[1] < 1 or motion.shape[2] != INTERX_MATRIX_TOTAL_DIM:
        return False

    root = motion[:, -1, :]
    tail_zero = np.allclose(root[:, 12:18], 0.0, atol=1e-6)
    middle_nonzero = not np.allclose(root[:, 6:12], 0.0, atol=1e-6)
    return tail_zero and middle_nonzero


def _repair_legacy_interx_matrix_layout(motion):
    """Convert the legacy buggy root packing into the expected per-person root9 format."""
    if not _is_legacy_interx_matrix_layout(motion):
        return motion

    fixed = motion.copy()
    root = fixed[:, -1, :]
    fixed_root = np.zeros_like(root)
    fixed_root[:, :INTERX_ROOT_BLOCK_DIM] = root[:, :INTERX_ROOT_BLOCK_DIM]
    fixed_root[:, INTERX_MATRIX_SINGLE_DIM:INTERX_MATRIX_SINGLE_DIM + INTERX_ROOT_BLOCK_DIM] = (
        root[:, INTERX_ROOT_BLOCK_DIM:INTERX_ROOT_BLOCK_DIM * 2]
    )
    fixed[:, -1, :] = fixed_root
    return fixed


def _prepare_processed_interx_motion(motion, data_rep, name=None):
    motion = motion.astype('float32', copy=False)
    data_rep = 'rot6d' if data_rep is None else str(data_rep).lower()

    if data_rep == 'noncanonical' and motion.ndim == 2:
        if motion.shape[1] != NONCANONICAL_TOTAL_FLAT_DIM:
            raise ValueError(
                f"Expected noncanonical flat dim {NONCANONICAL_TOTAL_FLAT_DIM}, got {motion.shape}"
                + (f" at key {name}" if name is not None else "")
            )
        return motion

    if motion.ndim != 3:
        raise ValueError(
            f"Expected motion shape [T, J, D], got {motion.shape}"
            + (f" at key {name}" if name is not None else "")
        )

    if data_rep == 'matrix':
        if motion.shape[2] != INTERX_MATRIX_TOTAL_DIM:
            raise ValueError(
                f"Expected matrix motion dim {INTERX_MATRIX_TOTAL_DIM}, got {motion.shape}"
                + (f" at key {name}" if name is not None else "")
            )
        motion = _repair_legacy_interx_matrix_layout(motion)

    return motion

class MotionDatasetV2HHI(data.Dataset):
    def __init__(self, opt, split_file, motion_file):
        self.opt = opt
        self.motion_rep = opt.motion_rep
        if opt.motion_rep == "global":
            motion_file = motion_file.replace('.h5', '_global.h5')
            self.normalize = True
            self.normalizer = InterxNormalizerTorch()

        self.data = []
        self.lengths = []
        self.num_person = 1
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        with h5py.File(motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            T,J,D = mf[id_list[0]][:].astype('float32').shape
            d =  D//2
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if motion.shape[0] < opt.window_size:
                        continue
                    
                    motion1, motion2 = motion[:, :, :d], motion[:, :, d:]
                    # self.lengths.append(motion1.shape[0] - opt.window_size)
                    self.data.append(motion1)
                    # self.lengths.append(motion2.shape[0] - opt.window_size)
                    self.data.append(motion2)
                except Exception as e: 
                    print(e)
                    pass

        # self.cumsum = np.cumsum([0] + self.lengths)
        self.indices = self._create_indices()
        print("Total number of motions {}, snippets {}".format(len(self.data), len(self.indices)))

    def _create_indices(self):
        # Create a list of tuples (sample_index, time_index) for data retrieval
        indices = []
        for i, sample in enumerate(self.data):
            for start_idx in range(0, len(sample) - self.opt.window_size + 1, self.opt.window_stride):
                indices.append((i, start_idx))
        return indices

    def inv_transform(self, data):
        return data

    def __len__(self):
        # return self.cumsum[-1]
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve a window of data based on the index
        sample_idx, time_idx = self.indices[idx]
        sample = self.data[sample_idx]
        motion = sample[time_idx:time_idx + self.opt.window_size]
        
        if self.motion_rep == "smpl":
            rot_6d = self.to_rot_6d(motion[:,:-1,:]).float()

            transl = to_torch(motion[:,-1,:])
            vel = transl[1:] - transl[:-1]
            vel = torch.cat([vel, torch.zeros(1, vel.shape[-1])], axis=0)
            transl_vel = torch.cat([transl, vel], axis=-1)

            motion = torch.cat([rot_6d, transl_vel.unsqueeze(1)], axis=1)
        
        elif self.motion_rep == "global":
            rot_6d = self.to_rot_6d(motion[:,:,-3:]).float()
            transl_vel = to_torch(motion[:,:,:-3])

            motion = torch.cat([transl_vel, rot_6d], axis=-1)
            
            if self.normalize:
                motion = self.normalizer.forward(motion)

        
        
        ## old 
        # if item != 0:
        #     motion_id = np.searchsorted(self.cumsum, item) - 1
        #     idx = item - self.cumsum[motion_id] - 1
        # else:
        #     motion_id = 0
        #     idx = 0
        # motion = self.data[motion_id][idx:idx+self.opt.window_size]
        # motion = self.to_rot_6d(motion).float()
        return motion #B,T,J,D

    def to_rot_6d(self, pose):
        pose = to_torch(pose)
        
        pose_all = []
        pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:,:,0:3])))
        
        ret = torch.cat(pose_all, dim=2)
        return ret

class MotionDatasetV2HHIProcessed(data.Dataset):
    """
    Dataloader variant for preprocessed H5 files that already store the target
    representation (e.g. rot6d, canonical). No axis-angle to rot6d conversion
    is performed; the loader only slices/pads as needed.
    """
    SUPPORTED_REPRESENTATIONS = ('rot6d', 'canonical', 'noncanonical', 'axis', 'quaternion', 'joint', 'matrix')

    def __init__(self, opt, split_file, motion_file, data_rep='rot6d'):
        self.opt = opt
        self.motion_rep = getattr(opt, 'motion_rep', 'smpl')
        self.data_rep = 'rot6d' if data_rep is None else str(data_rep).lower()
        if self.data_rep not in self.SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"Unsupported data representation {data_rep}. Choose from {self.SUPPORTED_REPRESENTATIONS}")

        self.data = []
        self.lengths = []
        self.num_person = 1
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        with h5py.File(motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = _prepare_processed_interx_motion(mf[name][:], self.data_rep, name=name)
                    if motion.shape[0] < opt.window_size:
                        continue
                    if self.data_rep == 'noncanonical' and motion.ndim == 2:
                        self.data.append(motion[:, :NONCANONICAL_PERSON_FLAT_DIM])
                        self.data.append(motion[:, NONCANONICAL_PERSON_FLAT_DIM:])
                    else:
                        if motion.shape[2] % 2 != 0:
                            raise ValueError(f"Expected even last dim for two-person split, got shape {motion.shape} at key {name}")
                        d = motion.shape[2] // 2
                        self.data.append(motion[:, :, :d])
                        self.data.append(motion[:, :, d:])
                except Exception as e:
                    print(e)
                    pass

        self.indices = self._create_indices()
        print("Total number of motions {}, snippets {}, rep {}".format(len(self.data), len(self.indices), self.data_rep))

    def _create_indices(self):
        indices = []
        for i, sample in enumerate(self.data):
            for start_idx in range(0, len(sample) - self.opt.window_size + 1, self.opt.window_stride):
                indices.append((i, start_idx))
        return indices

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, time_idx = self.indices[idx]
        sample = self.data[sample_idx]
        motion = sample[time_idx:time_idx + self.opt.window_size]
        motion = to_torch(motion).float()
        return motion

class MotionCaptionDatasetV2HHI(data.Dataset):
    def __init__(self, opt, split_file, motion_file, tokenizer):
        self.opt = opt
        self.motion_rep = opt.motion_rep
        self.tokenizer = tokenizer
        self.data = []
        self.num_person = 1

        if opt.motion_rep == "global":
            motion_file = motion_file.replace('.h5', '_global.h5')
            self.normalize = True
            self.normalizer = InterxNormalizerTorch()

        id_list = _load_split_ids(split_file)
        with h5py.File(motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if motion.shape[0] < opt.window_size:
                        continue
                    text_data = _load_text_data(opt.text_dir, name)
                    if not text_data:
                        continue

                    d = motion.shape[2] // 2
                    self.data.append({
                        'motion': motion[:, :, :d],
                        'paired_motion': motion[:, :, d:],
                        'text': text_data,
                        'name': name + '_p1',
                    })
                    self.data.append({
                        'motion': motion[:, :, d:],
                        'paired_motion': motion[:, :, :d],
                        'text': text_data,
                        'name': name + '_p2',
                    })
                except Exception as e:
                    print(e)
                    pass

        self.indices = self._create_indices()
        print("Total number of caption-aware motions {}, snippets {}".format(len(self.data), len(self.indices)))

    def _create_indices(self):
        indices = []
        for i, sample in enumerate(self.data):
            for start_idx in range(0, len(sample['motion']) - self.opt.window_size + 1, self.opt.window_stride):
                indices.append((i, start_idx))
        return indices

    def __len__(self):
        return len(self.indices)

    def _format_motion(self, motion):
        motion = to_torch(motion).float()

        if self.motion_rep == "smpl":
            rot_6d = self.to_rot_6d(motion[:, :-1, :]).float()
            transl = motion[:, -1, :]
            vel = transl[1:] - transl[:-1]
            vel = torch.cat([vel, torch.zeros(1, vel.shape[-1])], axis=0)
            transl_vel = torch.cat([transl, vel], axis=-1)
            motion = torch.cat([rot_6d, transl_vel.unsqueeze(1)], axis=1)
        elif self.motion_rep == "global":
            rot_6d = self.to_rot_6d(motion[:, :, -3:]).float()
            transl_vel = motion[:, :, :-3]
            motion = torch.cat([transl_vel, rot_6d], axis=-1)
            if self.normalize:
                motion = self.normalizer.forward(motion)

        return motion

    def __getitem__(self, idx):
        sample_idx, time_idx = self.indices[idx]
        sample = self.data[sample_idx]
        motion = self._format_motion(sample['motion'][time_idx:time_idx + self.opt.window_size])
        paired_motion = self._format_motion(sample['paired_motion'][time_idx:time_idx + self.opt.window_size])

        text_data = random.choice(sample['text'])
        token_ids, sent_len = self.tokenizer.encode_tokens(text_data['tokens'], self.opt.max_text_len)
        return motion, paired_motion, text_data['caption'], torch.tensor(token_ids, dtype=torch.long), sent_len

    def to_rot_6d(self, pose):
        pose = to_torch(pose)
        pose_all = [geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:, :, 0:3]))]
        return torch.cat(pose_all, dim=2)


class MotionCaptionDatasetV2HHIProcessed(data.Dataset):
    SUPPORTED_REPRESENTATIONS = ('rot6d', 'canonical', 'noncanonical', 'axis', 'quaternion', 'joint', 'matrix')

    def __init__(self, opt, split_file, motion_file, tokenizer, data_rep='rot6d'):
        self.opt = opt
        self.motion_rep = getattr(opt, 'motion_rep', 'smpl')
        self.data_rep = 'rot6d' if data_rep is None else str(data_rep).lower()
        self.tokenizer = tokenizer
        self.data = []
        if self.data_rep not in self.SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"Unsupported data representation {data_rep}. Choose from {self.SUPPORTED_REPRESENTATIONS}")

        id_list = _load_split_ids(split_file)
        with h5py.File(motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = _prepare_processed_interx_motion(mf[name][:], self.data_rep, name=name)
                    if motion.shape[0] < opt.window_size:
                        continue
                    text_data = _load_text_data(opt.text_dir, name)
                    if not text_data:
                        continue

                    if self.data_rep == 'noncanonical' and motion.ndim == 2:
                        self.data.append({
                            'motion': motion[:, :NONCANONICAL_PERSON_FLAT_DIM],
                            'paired_motion': motion[:, NONCANONICAL_PERSON_FLAT_DIM:],
                            'text': text_data,
                            'name': name + '_p1',
                        })
                        self.data.append({
                            'motion': motion[:, NONCANONICAL_PERSON_FLAT_DIM:],
                            'paired_motion': motion[:, :NONCANONICAL_PERSON_FLAT_DIM],
                            'text': text_data,
                            'name': name + '_p2',
                        })
                    else:
                        if motion.shape[2] % 2 != 0:
                            raise ValueError(f"Expected even last dim for two-person split, got shape {motion.shape} at key {name}")
                        d = motion.shape[2] // 2
                        self.data.append({
                            'motion': motion[:, :, :d],
                            'paired_motion': motion[:, :, d:],
                            'text': text_data,
                            'name': name + '_p1',
                        })
                        self.data.append({
                            'motion': motion[:, :, d:],
                            'paired_motion': motion[:, :, :d],
                            'text': text_data,
                            'name': name + '_p2',
                        })
                except Exception as e:
                    print(e)
                    pass

        self.indices = self._create_indices()
        print("Total number of caption-aware motions {}, snippets {}, rep {}".format(len(self.data), len(self.indices), self.data_rep))

    def _create_indices(self):
        indices = []
        for i, sample in enumerate(self.data):
            for start_idx in range(0, len(sample['motion']) - self.opt.window_size + 1, self.opt.window_stride):
                indices.append((i, start_idx))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, time_idx = self.indices[idx]
        sample = self.data[sample_idx]
        motion = sample['motion'][time_idx:time_idx + self.opt.window_size]
        paired_motion = sample['paired_motion'][time_idx:time_idx + self.opt.window_size]
        motion = to_torch(motion).float()
        paired_motion = to_torch(paired_motion).float()
        text_data = random.choice(sample['text'])
        token_ids, sent_len = self.tokenizer.encode_tokens(text_data['tokens'], self.opt.max_text_len)
        return motion, paired_motion, text_data['caption'], torch.tensor(token_ids, dtype=torch.long), sent_len


class Text2MotionDatasetV2HHI(data.Dataset):
    def __init__(self, opt, split_file, w_vectorizer, motion_file):
        self.opt = opt
        self.motion_rep = opt.motion_rep
        if opt.motion_rep == "global":
            motion_file = motion_file.replace('.h5', '_global.h5')
            self.normalize = True
            self.normalizer = InterxNormalizerTorch()

        self.w_vectorizer = w_vectorizer
        self.motion_file = motion_file
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.num_person = 2
        min_motion_len = 24 #30 if self.opt.dataset_name =='hhi_text' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                exit(-1)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e: 
                    print(e)
                    pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        
        ## Prepare motion
        if self.motion_rep == "smpl":
            rot_6d = self.to_rot_6d(motion[:,:-1,:]).float()

            transl = to_torch(motion[:,-1,:])
            vel = transl[1:] - transl[:-1]
            vel = torch.cat([vel, torch.zeros(1, vel.shape[-1])], axis=0)
            transl_vel_all = []
            for ii in range(self.num_person):
                transl_vel_all.append(torch.cat([transl[:,3*ii:3*ii+3], vel[:,3*ii:3*ii+3]], axis=1))
            transl_vel_all = torch.cat(transl_vel_all, axis=1)

            motion = torch.cat([rot_6d, transl_vel_all.unsqueeze(1)], axis=1)
        
        elif self.motion_rep == "global":
            motion1, motion2 = motion[..., :motion.shape[-1]//2], motion[..., motion.shape[-1]//2:]

            rot_6d1 = self.to_rot_6d(motion1[:,:,-3:]).float()
            transl_vel1 = to_torch(motion1[:,:,:-3])
            motion1 = torch.cat([transl_vel1, rot_6d1], axis=-1)

            rot_6d2 = self.to_rot_6d(motion2[:,:,-3:]).float()
            transl_vel2 = to_torch(motion2[:,:,:-3])
            motion2 = torch.cat([transl_vel2, rot_6d2], axis=-1)
            
            if self.normalize:
                motion1 = self.normalizer.forward(motion1)
                motion2 = self.normalizer.forward(motion2)

            motion = torch.cat([motion1, motion2], axis=-1)
            
        
        if m_length < self.max_motion_length:
            motion = torch.cat([motion,
                                torch.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                ], dim=0)
        else:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

    
    def to_rot_6d(self, pose):
        pose = to_torch(pose)

        pose_all = []
        pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:,:,0:3])))
        if self.motion_rep == "smpl":
            pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:,:,3:6])))
        
        ret = torch.cat(pose_all, dim=2) 
        return ret

class Text2MotionDatasetV2HHIProcessed(data.Dataset):
    """
    Text-conditioned loader for preprocessed motion features (e.g. rot6d).
    The H5 file is assumed to already store the desired representation; no
    conversion from axis-angle is applied.
    """
    SUPPORTED_REPRESENTATIONS = ('rot6d', 'canonical', 'noncanonical', 'axis', 'quaternion', 'joint', 'matrix')

    def __init__(self, opt, split_file, w_vectorizer, motion_file, data_rep='rot6d'):
        self.opt = opt
        self.motion_rep = getattr(opt, 'motion_rep', 'smpl')
        self.data_rep = 'rot6d' if data_rep is None else str(data_rep).lower()
        if self.data_rep not in self.SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"Unsupported data representation {data_rep}. Choose from {self.SUPPORTED_REPRESENTATIONS}")

        self.w_vectorizer = w_vectorizer
        self.motion_file = motion_file
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.num_person = 2
        min_motion_len = 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = _prepare_processed_interx_motion(mf[name][:], self.data_rep, name=name)
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                exit(-1)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e: 
                    print(e)
                    pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        if len(tokens) < self.opt.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        if m_length < self.max_motion_length:
            if motion.ndim == 3:
                pad = np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]), dtype=motion.dtype)
            elif motion.ndim == 2:
                pad = np.zeros((self.max_motion_length - m_length, motion.shape[1]), dtype=motion.dtype)
            else:
                raise ValueError(f"Unsupported motion rank {motion.ndim}")
            motion = np.concatenate([motion, pad], axis=0)
        else:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length

        motion = to_torch(motion).float()
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class Text2MotionDatasetHHI(data.Dataset):
    def __init__(self, opt, split_file, w_vectorizer, motion_file):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.motion_file = motion_file
        self.pointer = 0
        self.num_person = 2
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 30 if self.opt.dataset_name =='hhi_text' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    if len(motion) > self.max_motion_length:
                        motion = motion[:self.max_motion_length]
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                exit(-1)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e: 
                    print(e)
                    pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(['single', 'single', 'double'])
                else:
                    coin2 = 'single'
                if len_gap == 0 or (len_gap == 1 and coin2 == 'double'):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx+self.max_length]
                else:
                    if coin2 == 'single':
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'

            if coin2 == 'double':
                m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]

        if m_length >= self.max_motion_length:
            m_length = self.max_motion_length

        motion = self.to_rot_6d(motion).float()

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


    def to_rot_6d(self, data):
        joints3D = np.expand_dims(data[:,-1,:], axis=1)
        joints3D = to_torch(joints3D)
        ret_tr = joints3D[:, 0, :]

        pose = data[:,:-1,:]
        
        pose = to_torch(pose)
        pose_all = []
        for ii in range(self.num_person):
            pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:,:,3*ii:3*ii+3])))
        ret = torch.cat(pose_all, dim=2)
        padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
        for ii in range(self.num_person):
            padded_tr[:, 6*ii:6*ii+3] = ret_tr[:,3*ii:3*ii+3]
        ret = torch.cat((ret, padded_tr[:, None]), 1)
        T = ret.shape[0]
        ret = torch.reshape(ret, (T, -1))
        # ret = ret.permute(1, 2, 0).contiguous()
        return ret

class Text2MotionDatasetV2HHI_old(data.Dataset):
    def __init__(self, opt, split_file, w_vectorizer, motion_file):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.motion_file = motion_file
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.num_person = 2
        min_motion_len = 24 #30 if self.opt.dataset_name =='hhi_text' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                exit(-1)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e: 
                    print(e)
                    pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        else:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length
        motion = self.to_rot_6d(motion).float()
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

    def to_rot_6d(self, data):
        joints3D = np.expand_dims(data[:,-1,:], axis=1)
        joints3D = to_torch(joints3D)
        ret_tr = joints3D[:, 0, :]

        pose = data[:,:-1,:]
        
        pose = to_torch(pose)
        pose_all = []
        for ii in range(self.num_person):
            pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:,:,3*ii:3*ii+3])))
        ret = torch.cat(pose_all, dim=2)
        padded_tr = torch.zeros((ret.shape[0], ret.shape[2]), dtype=ret.dtype)
        for ii in range(self.num_person):
            padded_tr[:, 6*ii:6*ii+3] = ret_tr[:,3*ii:3*ii+3]
        ret = torch.cat((ret, padded_tr[:, None]), 1)
        T = ret.shape[0]
        ret = torch.reshape(ret, (T, -1))
        return ret
