# coding: utf-8

# 各种工具函数
import os
import copy
import time
import json
import math
import yaml
import torch
import pickle
import random
import cv2 as cv
import numpy as np
from rtree import index
from skimage import measure
from easydict import EasyDict 
from datetime import datetime
from scipy.spatial import cKDTree
from collections import defaultdict
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from skimage.morphology import skeletonize
from PIL import Image

from infer import infer_topo_patch_by_patch
import GTE_graph_utils.decoder as GTE_utils

def load_config(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def read_rgb(path):
    rgb = Image.open(path)
    return np.array(rgb)

# dataset related
def get_data_split(dataset='cityscale'):
    dataset = dataset.lower()
    assert dataset in ['cityscale', 'spacenet']
    
    if 'cityscale' == dataset:
        data_split = json.load(open('./data/cityscale/data_split.json'))
        valid_key = 'valid'
    elif 'spacenet' == dataset:
        data_split = json.load(open('./data/spacenet/data_split.json'))
        valid_key = 'validation'
    train_ids, valid_ids, test_ids = data_split['train'], data_split[valid_key], data_split['test']
    
    return train_ids, valid_ids, test_ids


def transform_coord(cfg, graph):
        '''coord transform for spacenet'''
        new_graph = {}
        for k, n in graph.items():
            src_r, src_c = k
            new_neighbors = []
            for nei in n:
                dst_r, dst_c = nei
                new_nei = (cfg.IMAGE_SIZE - dst_r, dst_c)
                new_neighbors.append(new_nei)
            new_k = (cfg.IMAGE_SIZE - src_r, src_c)
            new_graph[new_k] = new_neighbors
            
        return new_graph
    

def float2int_graph(graph):
    new_graph = {}
    for k, n in graph.items():
        new_k = (int(k[0]+0.5), int(k[1]+0.5))
        new_neighbors = []
        for nei in n:
            new_nei = (int(nei[0]+0.5), int(nei[1]+0.5))
            new_neighbors.append(new_nei)
        new_graph[new_k] = new_neighbors
        
    return new_graph

 
def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}

    for k in keys:
        if k == 'mask_targets':
            mask_targets = []
            for single_tgt in batch:
                mask_targets += single_tgt[k] # 不是stack而是cat
            collated[k] = mask_targets
        elif k == 'topo_targets':
            topo_targets = []
            for single_tgt in batch:
                topo_targets += single_tgt[k]
            collated[k] = topo_targets
        elif k == 'batched_topo_targets':   # for dev vis
            batched_topo_targets = []
            for single_tgt in batch:
                batched_topo_targets.append(single_tgt[k])
            collated[k] = batched_topo_targets
        else:
            collated[k] = torch.stack([item[k] for item in batch], dim=0)
            
    return collated


def get_patch_info_one_img(img_id, img_size, patch_size, sample_margin, patches_per_edge):
    patches = []
    min_location = sample_margin
    max_location = img_size - (sample_margin + patch_size)
    begin_location = np.linspace(start=min_location, stop=max_location, num=patches_per_edge, endpoint=True)
    begin_location = [round(x) for x in begin_location]
    for r in begin_location:
        for c in begin_location:
            patches.append(
                (img_id, (r, c), (r+patch_size, c+patch_size))
            )
            
    return patches


def get_batch_patch(img, patch_info):
    '''
    :param: 
        img: HWC
    '''
    patches = []
    for single_patch_info in patch_info:
        _, (start_r, start_c), (end_r, end_c) = single_patch_info
        single_patch = torch.tensor(img[start_r:end_r, start_c:end_c, :], dtype=torch.float32)
        patches.append(single_patch)
    batched_patches = torch.stack(patches).contiguous()
        
    return batched_patches


def get_output_dir_and_save_config(config, prefix, sepecified_dir=None):
    if sepecified_dir:
        output_dir = sepecified_dir
    else:
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(prefix, suffix)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    config_path = os.path.join(output_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(dict(config), f)
    
    return output_dir


def cal_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function {func.__name__} took {runtime:.3f} seconds to execute.")
        return result
    return wrapper


def detect_local_minima(arr, mask, threshold=0.5):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    idx = np.where((detected_minima & (mask > threshold)))
    scores = mask[idx]
    idx = np.column_stack(idx)
    	
    return idx, scores


# @cal_runtime
def nms_points(points, scores, radius, return_indices=False):
    # from SAM-Road
    # if score > 1.0, the point is forced to be kept regardless
    sorted_indices = np.argsort(scores)[::-1]   # 默认按值从小到大排序，改为从大到小，返回->值的索引
    sorted_points = points[sorted_indices, :]   # 点的rc坐标 组成的矩阵
    sorted_scores = scores[sorted_indices]      # 大于thr的分数 组成的列表
    kept = np.ones(sorted_indices.shape[0], dtype=bool)
    tree = cKDTree(sorted_points)
    for idx, p in enumerate(sorted_points):
        if not kept[idx]:
            continue
        neighbor_indices = tree.query_ball_point(p, r=radius)
        neighbor_scores = sorted_scores[neighbor_indices]
        
        keep_nbr = np.greater(neighbor_scores, 1.0)         # 等价于keep_nbr = neighbor_scores > 1.0       
        kept[neighbor_indices] = keep_nbr
        kept[idx] = True    
    if return_indices:
        return sorted_points[kept], sorted_indices[kept]
    else:
        return sorted_points[kept]
    

def merge_across_patch_predicts(cfg, raw_GTE, keypoints, thr=None):
    '''
    把每个 keypoint 在多个 patch 上对于同一邻接点的不完全重叠的预测合并为一个预测(形成一个邻接点)。
    是逐个kpt处理的

    :param raw_GTE: dict, 结构如下：   </br>
       { </br>              
        kpt_idx1: [ ([cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), [(cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), ... ] </br>
        kpt_idx2: [ ([cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), [(cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), ... ]  </br>
        ... </br>
        kpt_idxn: [ ([cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), [(cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), ... ]  </br>
    }
    
    :return GTE: dict: </br>
    {
        kpt_idx1: [[Δr1, Δc1], [Δr2, Δc2], ..., [Δrm, Δcm]] </br>
        kpt_idx2: [[Δr1, Δc1], [Δr2, Δc2], ..., [Δrn, Δcn]] </br>
        ... </br>
        kpt_idxn: [[Δr1, Δc1], [Δr2, Δc2], ..., [Δrp, Δcp]] </br>
    }</br>
    
    note: </br>
        1. m 不一定等于 n, p
    '''
    new_rel_GTE = {}
    for kpt_idx in range(len(keypoints)):
        all_pred_adj_infos = raw_GTE[kpt_idx]
        
        if not len(all_pred_adj_infos): # 原本就没有预测的邻接点，则填充一个空值
            new_rel_GTE[kpt_idx] = np.array([])
            continue
        
        all_pred_adj_validity   = np.row_stack(np.array([x[0] for x in all_pred_adj_infos]))
        all_pred_adj_rel_points = np.stack(np.array([x[1] for x in all_pred_adj_infos]), axis=0)
        
        # 剔除异常值
        # q1 = np.percentile(all_pred_adj_validity, 25, axis=0)
        # q3 = np.percentile(all_pred_adj_validity, 75, axis=0)
        
        # IQR = q3 - q1
        # lower_bound = q1 - 1.5 * IQR
        
        # all_pred_adj_validity = np.where((all_pred_adj_validity >= lower_bound), all_pred_adj_validity, np.nan)
        
        # abnormal_mask = np.isnan(all_pred_adj_validity)  # bool
        # mask_to_match = np.broadcast_to(np.expand_dims(abnormal_mask, axis=-1), abnormal_mask.shape+(2,)) # (num_patches, 10) -> (num_patches, 10, 2)
        
        # all_pred_adj_rel_points[mask_to_match] = np.nan
        all_pred_adj_validity = np.nanmean(all_pred_adj_validity, axis=0)   # --> (10,)
        all_pred_adj_rel_points = np.nanmean(all_pred_adj_rel_points, axis=0) # --> (10, 2)

        valid_thr = cfg.INFER.VALIDITY if (thr is None) else thr
        valid_mask = (all_pred_adj_validity >= valid_thr)
        
        all_pred_adj_validity = all_pred_adj_validity[valid_mask]
        all_pred_adj_rel_points = all_pred_adj_rel_points[valid_mask, :]
        
        if not len(all_pred_adj_rel_points):    # 原本有预测的邻接点，但是被滤除了，则也填充一个空值
            new_rel_GTE[kpt_idx] = np.array([])
        else:
            new_rel_GTE[kpt_idx] = all_pred_adj_rel_points
        
    return new_rel_GTE



def vis_GTE(cfg, GTE, keypoints, img_id, img, output_dir):
    rel_GTE = merge_across_patch_predicts(cfg, GTE, keypoints)
    vis_dir = os.path.join(output_dir, 'GTE_vis')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    rgb = img.copy()
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    vanilla_base_map = np.zeros_like(rgb)
    
    for keypoint_idx in range(len(keypoints)):
        r, c = keypoints[keypoint_idx]
        all_pred_adj_points = rel_GTE[keypoint_idx]
        for adj_pnt in all_pred_adj_points:
            delta_r, delta_c = adj_pnt
            adj_r, adj_c = int(r+cfg.NORM_D*delta_r), int(c+cfg.NORM_D*delta_c)
            cv.line(bgr, (c, r), (adj_c, adj_r), color=(255, 255, 255), thickness=1)
            cv.line(vanilla_base_map, (c, r), (adj_c, adj_r), color=(255, 255, 255), thickness=1) 
        cv.circle(bgr, (c, r), radius=1, color=(255, 0, 0), thickness=-1)
        cv.circle(vanilla_base_map, (c, r), radius=1, color=(255, 0, 0), thickness=-1)
    cv.imwrite(os.path.join(vis_dir, f'{img_id}_with_rgb.png'), bgr)
    cv.imwrite(os.path.join(vis_dir, f'{img_id}_no_rgb.png'), vanilla_base_map)
    
    

def cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return np.dot(v1, v2) / (norm_v1*norm_v2)


def find_best_candidte_strictly(
    cfg, 
    now_coord, 
    pred_coord, 
    length, 
    candidates, 
    keypoints, 
    rel_GTE, 
    min_distance, 
    angle_distance_weight
):
    '''
    :param now_coord: currently processing kpt's coord (r, c)
    :param pred_coord: pred coord calculated from the vert now kpt's delta_r, delta_c
    :param length: the distance between now kpt and the detected kpt(or edge_endpoint) around the pred_coord
    '''
    best_candidate = -1
    r, c = now_coord
    r1, c1 = pred_coord
    
    for candidate in candidates:
        if candidate >= len(keypoints):
            continue
        
        if candidate < len(keypoints):
            r_c, c_c = keypoints[candidate]
            
        d = np.linalg.norm(np.array([r_c-r1, c_c-c1]))     # 实际（邻接点）候选点和预测的（邻接点）之间的距离
        if d > length:
            continue
        v0 = np.array([r-r_c, c-c_c])    # 当前kpt到candidate点构成的边的向量
        
        min_sd = angle_distance_weight
        # b-->a
        candidate_pred_adj_points = rel_GTE[candidate]
        if len(candidate_pred_adj_points) == 0 :    # 当前候选点没有预测的邻接点
            continue
        for candidate_adj_pnt in candidate_pred_adj_points:
            candidate_delta_r, candidate_delta_c = candidate_adj_pnt
            vc = cfg.NORM_D * np.array([candidate_delta_r, candidate_delta_c])  # emit from candidate
            
            # 余弦距离 cosine distance = 1 - cosine similarity
            ad = 1.0 - cosine_similarity(v0, vc)    # [0, 2]
            ad *= angle_distance_weight
            
            if ad < min_sd:
                min_sd = ad

        d += min_sd  # 欧式距离+弧段距离，得到过去的距离
        
        # a-->b
        v1 = np.array([r_c-r, c_c-c])
        v2 = np.array([r1-r, c1-c])
        
        ad = 1.0 - cosine_similarity(v1, v2)
        d += ad*angle_distance_weight # 加上过来的（方向）距离，形成选择这个候选点的总距离

        if d < min_distance:
            min_distance = d
            best_candidate = candidate
            
    return best_candidate
        

def find_best_candidte_relaxedly(
    cfg, 
    now_coord, 
    pred_coord, 
    length, 
    candidates, 
    keypoints, 
    rel_GTE, 
    min_distance, 
    angle_distance_weight, 
    even_more=False, 
    edge_endpoints=None
):
    
    if not even_more:
        length = 0.5*length # 收紧距离约束
        
    best_candidate = -1
    r, c = now_coord
    r1, c1 = pred_coord
    
    for candidate in candidates:
        if not even_more:   # snap to keypoints
            if candidate >= len(keypoints):
                continue
            elif candidate < len(keypoints):
                r_c, c_c = keypoints[candidate]
        else:   # snap to edge_endpoints
            if edge_endpoints is None:
                continue
            if candidate < len(keypoints):
                continue
            elif candidate >= len(keypoints):
                r_c, c_c = edge_endpoints[candidate-len(keypoints)]
        
        d = np.linalg.norm(np.array([r_c, c_c]) - np.array(r1, c1))
        if d > length:
            continue
        
        # a-->b 
        v1 = np.array([r_c-r, c_c-c])
        v2 = np.array([r1-r, c1-c])
        ad = 1.0 - cosine_similarity(v1, v2)
        d += ad*angle_distance_weight     # 因为relax为只考虑一边（a-->b或b-->a）能满足要求，所以提高了角度带来的距离差异
        
        if d < min_distance:
            best_candidate = candidate
            min_distance = d
        # else:   # b-->a      
        #     d = np.linalg.norm(np.array([r_c, c_c]) - np.array(r1, c1)) # 欧氏距离
        #     min_sd = angle_distance_weight
        #     v0 = np.array([r-r_c, c-c_c])
        #     candidate_pred_adj_points = rel_GTE[candidate]
        #     for candidate_adj_pnt in candidate_pred_adj_points:
        #         candidate_delta_r, candidate_delta_c = candidate_adj_pnt
        #         vc = cfg.NORM_D * np.array([candidate_delta_r, candidate_delta_c])
        #         # 余弦距离 cosine distance = 1 - cosine similarity
        #         ad = 1.0 - cosine_similarity(v0, vc)
        #         ad *= angle_distance_weight   # 因为relax为只考虑一边（a-->b或b-->a）能满足要求，所以提高了角度带来的距离差异
        #         if ad < min_sd:
        #             min_sd = ad   
        #     d += min_sd
        #     if d < min_distance:
        #         best_candidate = candidate
        #         min_distance = d
                
    return best_candidate



# @cal_runtime
def GTE_decode(
    cfg, 
    output_dir, 
    img_id, 
    keypoints,
    rel_GTE,
    snap=True,
    snap_dist=15,
    drop=True,
    total_refine=True,
    angle_distance_weight=10
):
    
    '''  
    :param keypoints: </br>
    :param keypoints_rtree: </br>
    :param GTE: dict:{ </br>
        kpt_idx1: [[Δr1, Δc1], [Δr2, Δc2], ..., [Δrm, Δcm]] </br>
        kpt_idx2: [[Δr1, Δc1], [Δr2, Δc2], ..., [Δrn, Δcn]] </br>
        ... </br>
        kpt_idxn: [[Δr1, Δc1], [Δr2, Δc2], ..., [Δrp, Δcp]] </br>
    } </br>
    note: </br>
        1. m 不一定等于 n, p
    '''
    kpt_limit = 10000000
    cc = 0
    
    if snap == True:
        rgb = np.zeros((cfg.IMAGE_SIZE*4, cfg.IMAGE_SIZE*4, 3), dtype=np.uint8)
        rgb2 = np.zeros((cfg.IMAGE_SIZE*4, cfg.IMAGE_SIZE*4, 3), dtype=np.uint8)
    
    GTE_decode_output_dir = os.path.join(output_dir, 'decode_result')   # output_dir/decode_result/
    if not os.path.exists(GTE_decode_output_dir):
        os.makedirs(GTE_decode_output_dir)
    filename_prefix = os.path.join(GTE_decode_output_dir, f'{img_id}')
    
    keypoints_rtree = index.Index()
    for i in range(len(keypoints)):
        if cc > kpt_limit:
            break
        r, c = keypoints[i]
        keypoints_rtree.insert(i, (r, c, r, c))
        cc += 1
    
    # Step-2 Connect the vertices to build a graph. 
    # endpoint lookup 
    adj_dict = defaultdict(list)
    cc = 0
    
    for kpt_idx in range(len(keypoints)):
        if cc > kpt_limit:
            break
        r, c = keypoints[kpt_idx]
        now_coord = (r, c)
        pred_adj_points = rel_GTE[kpt_idx]   # [[Δr, Δc], [Δr, Δc], ...]
        
        if not len(pred_adj_points):
            continue
        
        for pnt in pred_adj_points:
            delat_r, delta_c = pnt
            r1, c1 = r+cfg.NORM_D*delat_r, c+cfg.NORM_D*delta_c
            # adj_r, adj_c = max(0, min(cfg.IMAGE_SIZE-1, adj_r)), max(0, min(cfg.IMAGE_SIZE-1, adj_c))
            skip = False
            pred_coord = (r1, c1)
            length = cfg.NORM_D * np.linalg.norm(np.array([delat_r, delta_c]))
            
            if snap == True:
                best_candidate = -1 
                min_distance = snap_dist
                
                query_box = (r1-20, c1-20, r1+20, c1+20)    # TODO 可以变换以改变性能
                candidates = list(keypoints_rtree.intersection(query_box))
        
                # pass 1: strict
                best_candidate = find_best_candidte_strictly(cfg, now_coord, pred_coord, length, candidates, keypoints, rel_GTE, min_distance, angle_distance_weight)
                
                # ***** pass 2 relax *****
                min_distance = snap_dist
                if best_candidate == -1:    # 说明pass1没有找到配对的点
                    best_candidate = find_best_candidte_relaxedly(cfg, now_coord, pred_coord, length, candidates, keypoints, rel_GTE, min_distance, angle_distance_weight*2, even_more=False)
                # ***** pass 3 more relax ***** 复用pass 2的代码，二者只是angledistance_weight不一样，pass2为原angledistance_weight的2倍，pass3保持不变
                # min_distance = snap_dist
                # if best_candidate == -1:
                #     best_candidate = find_best_candidte_relaxedly(cfg, now_coord, pred_coord, length, candidates, keypoints, rel_GTE, min_distance, angle_distance_weight, even_more=True, edge_endpoints=edge_endpoints)
                
                # finally 
                if best_candidate != -1:
                    if best_candidate < len(keypoints):
                        r1, c1 = keypoints[best_candidate]
                    # else:
                    #     r1, c1 = edge_endpoints[best_candidate-len(keypoints)]
                    #     r1, c1 = int(r1+0.5), int(c1+0.5)   # edge endpoint 是没有round to int的
                else:
                    skip = True
                      
            # gradually complete graph
            if skip == False or drop==False:
                p1 = (r1, c1)
                p2 = (r, c)
                if p1 != p2:
                    if p1 in adj_dict:
                        if p2 in adj_dict[p1]:
                            pass
                        else:
                            adj_dict[p1].append(p2)
                    else:   # p1 not in adj_dict
                        adj_dict[p1] = [p2] # insert
                        
                    if p2 in adj_dict:
                        if p1 in adj_dict[p2]:
                            pass
                        else:
                            adj_dict[p2].append(p1)
                    else: # p2 not in adj_dict
                        adj_dict[p2] = [p1]

                # vis edge
                color = (255, 255, 255)
                pt1, pt2 = p1[::-1], p2[::-1]   # to xy
                pt1, pt2 = [4*x for x in pt1], [4*x for x in pt2]
                if p1 != p2:
                    cv.line(rgb, pt1, pt2, color=color, thickness=1)    # white rgb仅展示图上应该有的点和边， 这里画边
        cc += 1  
    
    raw_graph = adj_dict
    adj_dict_for_refine = copy.deepcopy(adj_dict)
    # refine graph
    spurs_thr = 50      
    isolated_thr = 200
    
    if cfg.DATASET == 'spacenet':
        spurs_thr = 40
        isolated_thr = 100
    
    if total_refine:   
        graph = GTE_utils.graph_refine(adj_dict_for_refine, isolated_thr=isolated_thr, spurs_thr=spurs_thr)        
        rc = 100
        while rc > 0:
            graph, rc = GTE_utils.graph_refine_deloop(GTE_utils.graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr)) 
            # graph = GTE_utils.graph_shave(graph, spurs_thr = spurs_thr)
    else:   # 只是deloop
        # rc = 100
        # while rc > 0:
        # graph = adj_dict_for_refine
        graph, rc = GTE_utils.graph_refine_deloop(adj_dict_for_refine) 
        # graph = GTE_utils.graph_shave(graph, spurs_thr = spurs_thr)
        
    graph_refined = graph
    # vis point
    if snap == True:
        for kpt_idx in range(len(keypoints)):
            if cc > kpt_limit:
                break
            r, c = keypoints[kpt_idx]
            
            cv.circle(rgb, (c*4, r*4), radius=5, color=(0, 0, 255), thickness=-1)   # red  rgb仅展示图上应该有的点和边， 这里画点
            cc += 1
            
            d = 0 # degree of a keypoint
            if (r, c) not in graph:
                continue
            pred_adj_points = graph[(r, c)]   # [[Δr, Δc], [Δr, Δc], ...]
            if not len(pred_adj_points):
                continue
            for pnt in pred_adj_points:
                d += 1
            color = (0, 0, 255)  # d == 0 or d == 1
            if d == 2:
                color = (0, 255, 0)
            elif d == 3:
                color = (128, 128, 0)
            elif d >= 4:
                color = (255, 0, 0)
                
            cv.circle(rgb2, (c*4, r*4), color=color, radius=8, thickness=-1)    # 描述了每个点的度的情况
            
        # for i in range(len(edge_endpoints)):
        #     r, c = [int(x+0.5) for x in edge_endpoints[i]]
        #     cv.circle(rgb, (c*4, r*4), radius=3, color=(0, 255, 0), thickness=-1)   
        # green rgb仅展示图上应该有的点和边， 
        # 这里画由预测结果内插的点(这是原本sat2graph的写法)
        # 我们自己的方法只有在trace开启时才会插入edge_endpoints且这些点也进行了预测,跟keypoints无异
        
    cv.imwrite(filename_prefix+'_imagegraph.png', rgb)
    cv.imwrite(filename_prefix+'_intersection_node.png', rgb2)
    
    graph_without_tracing = graph
    if str(cfg.DATASET).lower() == 'spacenet':
        graph_without_tracing = transform_coord(cfg, graph_without_tracing)
    pickle.dump(graph_without_tracing, open(filename_prefix+"_graph_no_tracing.p","wb"))
    
    return rel_GTE, raw_graph, graph_refined

def get_candidates(extended_part_adj_dict):
    new_candidates = []
    for k in extended_part_adj_dict.keys():
        if (len(extended_part_adj_dict[k])==1):
            new_candidates.append(k)
            
    return new_candidates



def whether_merge_by_GTE(cfg, mode, now_coord, pred_delta, existed_pnt, rc_to_kpt_idx, rel_GTE):
    '''在追踪过程中，要弥合（到已有点）时需要看对方是否也预测当前点，也即双向奔赴了
        可以弥合: 
            1. decode后就有的点
            2. 上次追踪扩展出来的点
            3. 本次追踪已经添加的点（因为各个点到处理有先后顺序，而处理第一个点时 新添的点 可能会被作为 下一个被处理的点 要弥合的点）
    '''
    r, c = now_coord
    delta_r, delta_c = pred_delta
    pred_adj_r, pred_adj_c = r+cfg.NORM_D*delta_r, c+cfg.NORM_D*delta_c
    
    (existed_r, existed_c) = existed_pnt
    if (existed_r, existed_c) not in rc_to_kpt_idx:     
        # 既不在decode产生的图中，也不本次的候选点中（上次扩展完成后度为1的新添的点），那就只可能是本次扩展中已经新添进去的点（这个点根本就没有预测）
        # 离得足够近也行
        d = np.linalg.norm(np.array((existed_r, existed_c))- np.array((pred_adj_r, pred_adj_c)))
        if d < (cfg.INFER.BRIDGE_D / 2.0):
            return True
        else:
            return False
        
    kpt_idx = rc_to_kpt_idx[(existed_r, existed_c)]
    pred_adj_points = rel_GTE[kpt_idx] 
    
    if not len(pred_adj_points):    # XXX 对面那个点根本就没有预测点，被阈值过滤掉了，后续要对此依据不同阈值进行过滤，而非统一先过滤
        return False
    
    # emit to middle point
    for raw_coord in pred_adj_points:
        raw_delta_r, raw_delta_c = raw_coord
        raw_adj_r, raw_adj_c = existed_r+cfg.NORM_D*raw_delta_r, existed_c+cfg.NORM_D*raw_delta_c
        if mode == 'after_decode':
            d = np.linalg.norm(np.array((raw_adj_r, raw_adj_c)) - np.array((pred_adj_r, pred_adj_c)))  # 都预测了同一个中间点，才是最佳匹配
        else:
            d = np.linalg.norm(np.array((raw_adj_r, raw_adj_c)) - np.array((r, c)))
        ad = 1 - cosine_similarity(np.array((delta_r, delta_c)), -np.array((raw_delta_r, raw_delta_c)))    # 对面射过来的向量要反向 
        if (d < cfg.INFER.BRIDGE_D and ad < 0.02): # approximately 11.4 degree
            return True
            # best_matched_existed_point_idx = pnt_idx
    
    # emit to opposite
    # if best_matched_existed_point_idx == -1:
    for raw_coord in pred_adj_points:
        raw_delta_r, raw_delta_c = raw_coord
        raw_adj_r, raw_adj_c = existed_r+cfg.NORM_D*raw_delta_r, existed_c+cfg.NORM_D*raw_delta_c
        if mode == 'after_decode':
            d = np.linalg.norm(np.array((raw_adj_r, raw_adj_c))- np.array((pred_adj_r, pred_adj_c)))  # 都预测了同一个中间点，才是最佳匹配
        else:
            d = np.linalg.norm(np.array((raw_adj_r, raw_adj_c))- np.array((r, c)))
        if d < cfg.INFER.BRIDGE_D:
            # best_matched_existed_point_idx = pnt_idx
            return True
                    
    # return best_matched_existed_point_idx
    return False



def post_process_extended_GTE_to_grow_graph(
    cfg, 
    existed_adj_dict, 
    raw_rel_GTE, 
    raw_keypoints, 
    # raw_decode_graph_pnts_rtree, 
    existed_points, 
    existed_points_rtree, 
    to_extend_rel_GTE,
    extended_rel_GTE, 
    extended_candidates,
    candidates, 
    extended_part_adj_dict, 
    mode='after_decode',
): 
    extended_rc_to_kpt_idx = {}
    for cdt_idx in range(len(extended_candidates)):
        r, c = extended_candidates[cdt_idx]
        extended_rc_to_kpt_idx[(r, c)] = cdt_idx
        
    # 建立rc到最初的keypoints的映射，方便后续查找要弥合到的点（已有的） 的原始预测中是否有 要弥合过去的点（追踪预测的）
    raw_rc_to_kpt_idx = {}
    for kpt_idx in range(len(raw_keypoints)):
        r, c = raw_keypoints[kpt_idx]
        raw_rc_to_kpt_idx[(r, c)] = kpt_idx
    
    raw_keypoints_rtree = index.Index()
    for i, pnt in enumerate(raw_keypoints):
        r, c = pnt
        raw_keypoints_rtree.insert(i, (r, c, r, c))
        
    for idx in range(len(candidates)):
        r, c = candidates[idx]
        now_coord = (r, c)
        pred_adj_points = to_extend_rel_GTE[idx]   # [[Δr, Δc], [Δr, Δc], ...]
        
        if not len(pred_adj_points):
            continue
        
        for delta_coord in pred_adj_points:
            delta_r, delta_c = delta_coord
            adj_r, adj_c = r+cfg.NORM_D*delta_r, c+cfg.NORM_D*delta_c
            
            edge_existed = False
            pred_direction = np.array((delta_r, delta_c))
            for nei in existed_adj_dict[now_coord]:
                existed_direction = np.array(nei) - np.array(now_coord)
                ad = 1 - cosine_similarity(pred_direction, existed_direction)
                if ad < 0.02:   # 大概11.4°
                    edge_existed = True
            if edge_existed:    # 有一条同方向的边了，跳过
                continue
            
            # 是否弥合已有点
        #     if mode == 'after_decode': 
        #         assert raw_decode_graph_pnts_rtree is not None
        #         existed_points_idx = list(existed_points_rtree.intersection((adj_r-cfg.INFER.SERCH_D, 
        #                                                                     adj_c-cfg.INFER.SERCH_D, 
        #                                                                     adj_r+cfg.INFER.SERCH_D, 
        #                                                                     adj_c+cfg.INFER.SERCH_D)))
        # else:
            existed_points_idx = list(existed_points_rtree.intersection((adj_r-cfg.INFER.BRIDGE_D, 
                                                                        adj_c-cfg.INFER.BRIDGE_D, 
                                                                        adj_r+cfg.INFER.BRIDGE_D, 
                                                                        adj_c+cfg.INFER.BRIDGE_D)))
            if len(existed_points_idx):
                best_matched_existed_point_idx = -1
                
                for pnt_idx in existed_points_idx:
                    existed_r, existed_c = existed_points[pnt_idx]

                    if (existed_r, existed_c) == now_coord:  # 排除自身
                        continue
                    
                    if (existed_r, existed_c) not in raw_rc_to_kpt_idx.keys():  # 有的点可能是新添的，原始预测里面没有
                        rc_to_kpt_idx = extended_rc_to_kpt_idx
                        valid_GTE = extended_rel_GTE
                    else:
                        rc_to_kpt_idx = raw_rc_to_kpt_idx
                        valid_GTE = raw_rel_GTE
                    
                    if whether_merge_by_GTE(
                        cfg=cfg,
                        mode=mode, 
                        now_coord=now_coord,
                        pred_delta=delta_coord, 
                        existed_pnt=existed_points[pnt_idx], 
                        rc_to_kpt_idx=rc_to_kpt_idx, 
                        rel_GTE=valid_GTE
                    ):
                        best_matched_existed_point_idx = pnt_idx
                        break
                                
                if best_matched_existed_point_idx != -1:
                
                    adj_r, adj_c = existed_points[best_matched_existed_point_idx]
                    next_coord = (adj_r, adj_c)
                    
                    # 有一条同方向的边了，把那条短边去掉
                    for nei in existed_adj_dict[(adj_r, adj_c)]:
                        pred_direction = np.array(now_coord) - np.array(next_coord)
                        existed_direction = np.array(nei) - np.array(next_coord)
                        ad = 1 - cosine_similarity(pred_direction, existed_direction)
                        if ad < 0.04:   # 大概11.4°
                            existed_adj_dict[next_coord].remove(nei)
                            if next_coord in existed_adj_dict[nei]:
                                existed_adj_dict[nei].remove(next_coord)

                    if next_coord not in existed_adj_dict[now_coord]:
                        existed_adj_dict[now_coord].append(next_coord)    # add edge
                    if now_coord not in existed_adj_dict[next_coord]:
                        existed_adj_dict[next_coord].append(now_coord)    # add edge
                        
                    extended_part_adj_dict[now_coord].append(next_coord)
                    continue    
            
            # 是否弥合 之前被分割出来了但又被解码算法过滤掉了的点
            if mode == 'after_decode': 
                raw_keypoints_idx = list(
                    raw_keypoints_rtree.intersection(
                        (adj_r-cfg.INFER.BRIDGE_D,
                         adj_c-cfg.INFER.BRIDGE_D,
                         adj_r+cfg.INFER.BRIDGE_D,
                         adj_c+cfg.INFER.BRIDGE_D)
                    )
                )
            else:
                raw_keypoints_idx = []
            if len(raw_keypoints_idx):
                adj_r, adj_c = raw_keypoints[raw_keypoints_idx[0]]    # XXX 随便找一个可选点作要弥合的点
                next_coord = (adj_r, adj_c)
                if next_coord not in existed_adj_dict[now_coord]:
                    existed_adj_dict[now_coord].append(next_coord)    # add edge
                
                if next_coord not in existed_adj_dict:
                    existed_adj_dict[next_coord] = []   # add vertex
                    existed_adj_dict[next_coord].append(now_coord)    # add edge
                    existed_points_rtree.insert(len(existed_points), (adj_r, adj_c, adj_r, adj_c))
                    existed_points.append(next_coord)   # add vertex
                    
                extended_part_adj_dict[next_coord].append(now_coord)    # default dict
                extended_part_adj_dict[now_coord].append(next_coord)
                
            else:   # 无需弥合，新添点
                # if mode == 'after_decode': 
                #     assert  raw_decode_graph_pnts_rtree is not None
                #     raw_points_idx = list(raw_decode_graph_pnts_rtree.intersection((adj_r-10, adj_c-10, adj_r+10, adj_c+10)))
                #     if len(raw_points_idx):     # 说明这个点之前就有来着，后来被refine掉了，就不要添进来，需要跳过
                #         continue
                adj_r, adj_c = int(adj_r+0.5), int(adj_c+0.5)
                next_coord = (adj_r, adj_c)
                existed_adj_dict[next_coord] = []
                existed_adj_dict[next_coord].append(now_coord)    # add edge
                existed_points_rtree.insert(len(existed_points), (adj_r, adj_c, adj_r, adj_c))
                existed_points.append(next_coord)   # add vertex
                existed_adj_dict[now_coord].append(next_coord)    # add edge
                
                extended_part_adj_dict[next_coord].append(now_coord)    # default dict
                extended_part_adj_dict[now_coord].append(next_coord) 
    
    return existed_adj_dict, extended_part_adj_dict



# @cal_runtime
def graph_growing(
    cfg, 
    model, 
    raw_rel_GTE, 
    raw_keypoints, 
    # raw_decode_graph, 
    adj_dict, 
    starting_points, 
    all_backbone_out, 
    all_patch_info, 
    mode='after_decode', 
    total_refine=True
):
    '''
        :param:
            image_embeddings: [B, C, H, W], H=W=cfg.PATCH_SIZE//16
    '''
    initial_candidates = []     
    existed_points = []
    existed_points_rtree = index.Index()
    if mode == 'after_decode':
        assert adj_dict is not None
        for i, cdt in enumerate(adj_dict.keys()):  # make sure the adj_dict is ordered
            r, c = cdt
            existed_points.append((r, c))
            existed_points_rtree.insert(i, (r, c, r, c))

            if len(adj_dict[cdt]) == 1:     # only grow from points whose degree is 1
                initial_candidates.append(cdt)  # push all initial candidates)
    
    elif mode == 'direct':
        assert starting_points is not None
        for i, cdt in enumerate(starting_points):
            r, c = cdt
            existed_points.append((r, c))
            existed_points_rtree.insert(i, (r, c, r, c))
            initial_candidates.append((r, c))  # push all initial candidates
    
    original_len = len(adj_dict.keys()) if mode == 'after_decode' else len(starting_points)
    # print(f"======> 找到{len(initial_candidates)}个迭代起点")
    
    # trace
    adj_dict = adj_dict if (adj_dict is not None) else defaultdict(list)
    extended_part_adj_dict = defaultdict(list)
    
    candidates = np.array(initial_candidates)
    
    # raw_decode_graph_pnts = []
    # if raw_decode_graph is not None:    # 开启了decode
    #     raw_decode_graph_pnts_rtree = index.Index()     # points in decoded graph before refine
    #     for i, cdt in enumerate(raw_decode_graph.keys()): 
    #         r, c = cdt
    #         # raw_decode_graph_pnts.append((r, c))
    #         raw_decode_graph_pnts_rtree.insert(i, (r, c, r, c))
    # else:
    #     raw_decode_graph_pnts_rtree = None
    

    
    spurs_thr = 50
    isolated_thr = 200
    
    if cfg.DATASET == 'spacenet':
        spurs_thr = 30
        isolated_thr = 100
        
    growing_times = 0
    extended_rel_GTE = {}
    extended_candidates = []
    while (len(candidates) and (growing_times < cfg.INFER.EXTEND_TIMES)):
        to_extend_GTE = infer_topo_patch_by_patch(   # extended_GTE中的key为idx， 该idx用来在candidates中找到相应点(的坐标)
            model=model,
            all_patch_info=all_patch_info,
            batch_size=cfg.INFER.INFER_BATCH_SIZE,
            all_backbone_out=all_backbone_out,
            keypoints=candidates
        )
        
        to_extend_rel_GTE = merge_across_patch_predicts(cfg, to_extend_GTE, candidates, thr=cfg.INFER.TRACE_THR)

        original_extended_candidates_len = len(extended_candidates)
        extended_candidates += candidates.tolist()
        for idx in range(len(candidates)):  # 多次的预测结果合并
            extended_rel_GTE[original_extended_candidates_len+idx] = to_extend_rel_GTE[idx]

        adj_dict, extended_part_adj_dict = post_process_extended_GTE_to_grow_graph(
            cfg=cfg, 
            existed_adj_dict=adj_dict, 
            raw_rel_GTE=raw_rel_GTE,
            raw_keypoints=raw_keypoints, 
            # raw_decode_graph_pnts_rtree=raw_decode_graph_pnts_rtree, 
            existed_points=existed_points, 
            existed_points_rtree=existed_points_rtree, 
            to_extend_rel_GTE=to_extend_rel_GTE,    # 单次基于候选点的预测，想要扩展的部分
            extended_rel_GTE=extended_rel_GTE,  # 根据此预测，上次就已经做过扩展
            extended_candidates=extended_candidates,    # 上次扩展用的候选点
            candidates=candidates, 
            extended_part_adj_dict=extended_part_adj_dict,  # 每次追踪扩展的部分都累积在这里
            mode=mode
        )
        
        for k, v in adj_dict.items():   # 移除可能的 邻接点有自己的情况，防止后续refine时出现除0错误
            if k in v:
                v.remove(k)
            adj_dict[k] = v
            
        rc = 100
        while rc > 0:
            adj_dict, rc = GTE_utils.graph_refine_deloop(GTE_utils.graph_refine(adj_dict, isolated_thr=isolated_thr, spurs_thr=spurs_thr))
        
        existed_points = []
        existed_points_rtree = index.Index()
        for i, cdt in enumerate(adj_dict.keys()):  # make sure the adj_dict is ordered
            r, c = cdt
            existed_points.append((r, c))
            existed_points_rtree.insert(i, (r, c, r, c))
        
        growing_times += 1
        candidates = np.array(get_candidates(adj_dict))
    # end while
    # refine graph
    graph = adj_dict
    # if total_refine and cfg.DATASET == 'cityscale':     # cityscale更加规整，最后refine一次就好
    #     graph = GTE_utils.graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    #     rc = 100
    #     while rc > 0:
    #         graph, rc = GTE_utils.graph_refine_deloop(GTE_utils.graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))  
            # graph, rc = GTE_utils.graph_deduplication(GTE_utils.graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))  
        
    # graph = GTE_utils.graph_shave(graph, spurs_thr=spurs_thr)
        
    graph_refined = graph
     
    return graph_refined, extended_part_adj_dict
    