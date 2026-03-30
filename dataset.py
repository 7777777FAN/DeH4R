# coding: utf-8

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import utils
from label_generator import LabelGnerator
import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils import load_config, collate_fn
import h5py

class DeH4R_Dataset(Dataset):
    def __init__(self, cfg, is_train=True, dev_run=False):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        self.dev_run = dev_run
        
        self.dataset = cfg.DATASET.lower()
        assert self.dataset in ['cityscale', 'spacenet', 'globalscale']
        
        if 'cityscale' == self.dataset:
            self.IMAGE_SIZE = 2048
            self.SAMPLE_MARGIN = 64
            
            self.rgb_pattern = './data/cityscale/20cities/region_{}_sat.png'
            self.keypoint_mask_pattern = './data/cityscale/keypoint_mask/region_{}.png'
            self.samplepoint_mask_pattern = './data/cityscale/samplepoint_mask/region_{}.png'
            self.road_mask_pattern = './data/cityscale/road_mask/region_{}.png'
            self.graph_pattern = './data/cityscale/20cities/region_{}_refine_gt_graph.p'
        
            train_ids, val_ids, test_ids = utils.get_data_split(dataset=self.dataset)
            
        elif 'globalscale' == self.dataset:
            self.IMAGE_SIZE = 2048
            self.SAMPLE_MARGIN = 64
            self.hdf5_file_path = './data/globalscale/data_zip.h5'
            
            # self.rgb_pattern =  './data/globalscale/all/region_{}_sat.png'
            # self.keypoint_mask_pattern = './data/globalscale/keypoint_mask/region_{}.png'
            # self.samplepoint_mask_pattern = './data/globalscale/samplepoint_mask/region_{}.png'
            # self.road_mask_pattern = './data/globalscale/road_mask/region_{}.png'
            # self.graph_pattern =  './data/globalscale/all/region_{}_refine_gt_graph.p'

            train_ids, val_ids, test_ids, test_ids_out = utils.globalscale_data_partition()
            
        elif 'spacenet' == self.dataset:
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0
            self.rgb_pattern = './data/spacenet/RGB_1.0_meter/{}__rgb.png'
            self.keypoint_mask_pattern = './data/spacenet/keypoint_mask/{}.png'
            self.samplepoint_mask_pattern = './data/spacenet/samplepoint_mask/{}.png'
            self.road_mask_pattern = './data/spacenet/road_mask/{}.png'
            self.graph_pattern = './data/spacenet/RGB_1.0_meter/{}__gt_graph_dense.p'
            
            train_ids, val_ids, test_ids = utils.get_data_split(dataset=self.dataset)
            
            
        # train_ids += val_ids  # differ from sam-road
        # self.tile_ids = train_ids if is_train else test_ids
        
        self.tile_ids = train_ids if is_train else val_ids
        
        if dev_run:
            self.tile_ids = self.tile_ids[:4]
            
        self.rgbs = []
        self.keypoint_masks = []
        self.samplepoint_masks = []
        self.road_masks = []
        self.graph_label_generators = []
        
        self.skipped_tile_ids = []
        self.eval_patches = []
        if not is_train:
            eval_patches_per_edge = math.ceil((self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.cfg.PATCH_SIZE)  # 这句代码使得在首尾抠除margin大小后，eval时只有除不尽时才会有重叠，不够的部分会靠均匀地重叠来达成
            for idx, _ in enumerate(self.tile_ids):
                self.eval_patches += utils.get_patch_info_one_img(idx, self.IMAGE_SIZE, self.cfg.PATCH_SIZE, self.SAMPLE_MARGIN, eval_patches_per_edge)
        
        self._load_data()
        # assert False
    
    def _load_data(self):
        print(" ========= Loading Data =========")
        if 'globalscale' == self.dataset:
            self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        
        for id in tqdm(self.tile_ids):
            # rgb = utils.read_rgb(self.rgb_pattern.format(id))
            # keypoint_mask = utils.read_rgb(self.keypoint_mask_pattern.format(id))
            # samplepoint_mask = utils.read_rgb(self.samplepoint_mask_pattern.format(id))
            # road_mask = utils.read_rgb(self.road_mask_pattern.format(id))
            # graph = pickle.load(open(self.graph_pattern.format(id), 'rb'))
            
            # self.rgbs.append(rgb)
            # self.keypoint_masks.append(keypoint_mask)
            # self.samplepoint_masks.append(samplepoint_mask)
            # self.road_masks.append(road_mask)
        
            # generator = LabelGnerator(self.cfg, graph)
            # self.graph_label_generators.append(generator)
            
            self.rgbs.append(id)    # 只添加样本名称而不是真正的数据，使用lazy load
            
    def close(self):
        if 'globalscale' == self.dataset:
            self.hdf5_file.close()
    
    def __len__(self):
        if self.is_train:
            if 'cityscale' == self.dataset:
                if self.dev_run:
                    # return 512
                    return max(1, int(self.IMAGE_SIZE / self.cfg.PATCH_SIZE)) ** 2 * 2500
                else:
                    return max(1, int(self.IMAGE_SIZE / self.cfg.PATCH_SIZE)) ** 2 * 2500  
            elif 'globalscale' == self.dataset:
                if self.dev_run:
                    return 512
                else:
                    # num_patches_per_image = max(1, int(self.IMAGE_SIZE / self.config.PATCH_SIZE)) ** 2
                    # return  len(self.tile_ids) * num_patches_per_image
                    return max(1, int(self.IMAGE_SIZE / self.cfg.PATCH_SIZE)) ** 2 * 38000  
            elif 'spacenet' == self.dataset:
                if self.dev_run:
                    return 2048
                else:
                    return 84667 
        else:
            return len(self.eval_patches)
    
    
    def __getitem__(self, idx):
        # 输入X：
        #   ① RGB ② graph_keypoints_coord
        # 标签Y：
        #   ① keypoint_mask ② gt_adj_prob_and_coord ③ pad_mask ④ samplepoint_mask ⑤ road_mask
        min_location = self.SAMPLE_MARGIN
        max_location = self.IMAGE_SIZE - (self.SAMPLE_MARGIN + self.cfg.PATCH_SIZE)
        if self.is_train:
            img_idx = -1
            while (-1 == img_idx) or (img_idx in self.skipped_tile_ids):
                img_idx = np.random.randint(low=0, high=len(self.rgbs))
            begin_r = np.random.randint(low=min_location, high=max_location+1)
            begin_c = np.random.randint(low=min_location, high=max_location+1)
            end_r, end_c = begin_r+self.cfg.PATCH_SIZE, begin_c+self.cfg.PATCH_SIZE
        else:
            img_idx, (begin_r, begin_c), (end_r, end_c) = self.eval_patches[idx]
            
        rot_times = 0
        if self.is_train:
            rot_times = np.random.randint(0, 4)
            # rot_times = 3
        patch_info = (begin_r, begin_c), (end_r, end_c)
        keypoints, next_nodes_classes_and_coords, valid, all_null_flag, final_img_idx, final_patch_info = self.fetch_graph_data_for_all_null_condition(img_idx, patch_info, rot_times)
        
        next_nodes_calsses = [ns[0] if 0==int(ns[0][0]) else [] for ns in next_nodes_classes_and_coords]
        next_nodes_coords = [ns[1] if 0==int(ns[0][0]) else [] for ns in next_nodes_classes_and_coords]
        
        (begin_r, begin_c), (end_r, end_c) = final_patch_info
        
        img_name = self.rgbs[final_img_idx]
        
        if 'globalscale' == self.dataset:
            rgb = self.hdf5_file[f'rgbs/{img_name}'][:]
            keypoint_mask = self.hdf5_file[f'keypoint_masks/{img_name}'][:]
            samplepoint_mask = self.hdf5_file[f'samplepoint_masks/{img_name}'][:]
            road_mask = self.hdf5_file[f'road_masks/{img_name}'][:]
        else:
            rgb = utils.read_rgb(self.rgb_pattern.format(img_name))
            keypoint_mask = utils.read_rgb(self.keypoint_mask_pattern.format(img_name))
            samplepoint_mask = utils.read_rgb(self.samplepoint_mask_pattern.format(img_name))
            road_mask = utils.read_rgb(self.road_mask_pattern.format(img_name))
            
        
        rgb_patch = rgb[begin_r:end_r, begin_c:end_c]
        keypoints_mask_patch = keypoint_mask[begin_r:end_r, begin_c:end_c]
        samplepoints_mask_patch = samplepoint_mask[begin_r:end_r, begin_c:end_c]
        road_mask_patch = road_mask[begin_r:end_r, begin_c:end_c]
        
        # rgb_patch = self.rgbs[final_img_idx][begin_r:end_r, begin_c:end_c]
        # keypoints_mask_patch = self.keypoint_masks[final_img_idx][begin_r:end_r, begin_c:end_c]
        # samplepoints_mask_patch = self.samplepoint_masks[final_img_idx][begin_r:end_r, begin_c:end_c]
        # road_mask_patch = self.road_masks[final_img_idx][begin_r:end_r, begin_c:end_c]
        
        # aug
        if self.is_train:
            rgb_patch = np.rot90(rgb_patch, rot_times, axes=[0, 1]).copy()
            keypoints_mask_patch = np.rot90(keypoints_mask_patch, rot_times, axes=[0, 1]).copy()
            samplepoints_mask_patch = np.rot90(samplepoints_mask_patch, rot_times, axes=[0, 1]).copy()
            road_mask_patch = np.rot90(road_mask_patch, rot_times, axes=[0, 1]).copy()
        
        # assemble 
        mask_targets = [
            {
                'keypoints_mask': torch.tensor(keypoints_mask_patch, dtype=torch.float32) / 255.0,
                'samplepoints_mask': torch.tensor(samplepoints_mask_patch, dtype=torch.float32) / 255.0,
                'road_mask': torch.tensor(road_mask_patch, dtype=torch.float32) / 255.0,
            }
        ]
        
        topo_targets = [
            {
                'labels': torch.tensor(single_kpt_next_nodes_classes, dtype=torch.uint8), 
                'coords': torch.tensor(single_kpt_next_nodes_coords, dtype=torch.float32),
            }   
            for single_kpt_next_nodes_classes, single_kpt_next_nodes_coords in zip(next_nodes_calsses, next_nodes_coords)
        ]
        
        # for vis
        if self.cfg.VIS_TRAIN:
            batched_topo_targets = [
                {
                    'labels': torch.tensor(single_kpt_next_nodes_classes, dtype=torch.uint8), 
                    'coords': torch.tensor(single_kpt_next_nodes_coords, dtype=torch.float32),
                }   
                for single_kpt_next_nodes_classes, single_kpt_next_nodes_coords in zip(next_nodes_calsses, next_nodes_coords)
            ]
        
        return {
            # input
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
            'keypoints': torch.tensor(keypoints, dtype=torch.float32),
            # label
            'mask_targets': mask_targets,
            'topo_targets': topo_targets,
            'batched_topo_targets': None if not self.cfg.VIS_TRAIN else batched_topo_targets,
            # pad_mask
            'valid_mask': torch.tensor(valid, dtype=torch.bool)
        }
        
        
    def fetch_graph_data_for_all_null_condition(self, img_idx, patch_info, rot_times):
        img_name = self.rgbs[img_idx]
        if 'globalscale' == self.dataset:
            graph_data = self.hdf5_file[f'graphs/{img_name}'][:]
            graph = pickle.loads(graph_data.tobytes())
        else:
            graph = pickle.load(open(self.graph_pattern.format(img_name), 'rb'))
        generator = LabelGnerator(self.cfg, graph)
        keypoints, next_nodes_classes_and_coords, valid, all_null_flag = generator.sample_patch(patch_info, rot_times)
        # keypoints, next_nodes_classes_and_coords, valid, all_null_flag = self.graph_label_generators[img_idx].sample_patch(patch_info, rot_times)
        if all_null_flag and self.is_train:
            min_location = self.SAMPLE_MARGIN
            max_location = self.IMAGE_SIZE - (self.SAMPLE_MARGIN + self.cfg.PATCH_SIZE)
            img_idx = -1
            while (-1 == img_idx) or (img_idx in self.skipped_tile_ids):
                img_idx = np.random.randint(low=0, high=len(self.rgbs))
            begin_r = np.random.randint(low=min_location, high=max_location+1)
            begin_c = np.random.randint(low=min_location, high=max_location+1)
            end_r, end_c = begin_r+self.cfg.PATCH_SIZE, begin_c+self.cfg.PATCH_SIZE
            patch_info = (begin_r, begin_c), (end_r, end_c)
            
            return self.fetch_graph_data_for_all_null_condition(img_idx, patch_info, rot_times)
                
        return keypoints, next_nodes_classes_and_coords, valid, all_null_flag, img_idx, patch_info
        
        

class DeH4R_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def prepare_data(self):
        self.train_dataset = DeH4R_Dataset(cfg=self.cfg, is_train=True, dev_run=self.cfg.dev_run)
        self.val_dataset = DeH4R_Dataset(cfg=self.cfg, is_train=False, dev_run=self.cfg.dev_run)
    
    def setup(self, stage=None):
        self.train_dataset = self.train_dataset
        self.val_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.BATCH_SIZE, 
            shuffle=True, 
            pin_memory=True,
            num_workers=self.cfg.NUM_WORKERS,
            collate_fn=collate_fn
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=self.cfg.NUM_WORKERS,
            collate_fn=collate_fn
        )
        