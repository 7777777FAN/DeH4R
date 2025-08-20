# coding: utf-8

import pickle, json
import math
import cv2 as cv
import os
import numpy as np
import math
from tqdm import tqdm
from argparse import ArgumentParser



D_THR = 30
ROAD_WIDTH = 3
R = 3
THETA_THR = 15 # 度数，任意一个点的邻接边组成的任意一组夹角大于这个阈值则认为是大曲率点


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def kick_off_invalid_edges(graph):
    new_graph = {}
    for k, n in graph.items():
        src_r, src_c = k
        neighbors = []
        for nei in n:
            dst_r, dst_c = nei
            delta_r = dst_r - src_r
            delta_c = dst_c - src_c
            d = math.sqrt(delta_r**2 + delta_c**2)
            if d < D_THR:
                neighbors.append(nei)
        new_graph[k] = neighbors
        
    return new_graph   


def cos_similarity(v0 ,v1):
    norm_0 = np.linalg.norm(v0)
    norm_1 = np.linalg.norm(v1)
    if (norm_0 < 1e-5)or (norm_1 < 1e-5):
        return 1
    return np.dot(v0, v1) / (norm_0*norm_1)


def transform_coord(graph):
    '''coord transform for spacenet'''
    new_graph = {}
    for k, n in graph.items():
        src_r, src_c = k
        new_neighbors = []
        for nei in n:
            dst_r, dst_c = nei
            new_nei = (IMAGE_SIZE - dst_r, dst_c)
            new_neighbors.append(new_nei)
        new_k = (IMAGE_SIZE - src_r, src_c)
        new_graph[new_k] = new_neighbors
        
    return new_graph
    
    
def main():
    data_split_path = os.path.join(ROOT_DIR, 'data_split.json')
    data_split = json.load(open(data_split_path, 'rb'))
    all_names = []
    for split, names in data_split.items():
        all_names.extend(names)
    for i in tqdm(all_names):
        graph = pickle.load(open(GRAPH_PATTERN.format(i), 'rb'))
        # graph = kick_off_invalid_edges(graph)
        graph = transform_coord(graph)
        keypoint_map = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for node, neis in graph.items():
            node_y, node_x = node
            if len(neis) != 2:  # 度不为2
                cv.circle(keypoint_map, (node_x, node_y), radius=R, color=255, thickness=-1)
            
            else:# 度为2，但是曲率够大
                vecs = []
                for nei in neis:
                    nei_y, nei_x = nei
                    v = np.array([nei_x-node_x, nei_y-node_y])
                    vecs.append(v)
                
                v0, v1 = vecs
                if abs(cos_similarity(v0, v1)) < math.cos(THETA_THR*math.pi/180):
                    cv.circle(keypoint_map, (node_x, node_y), radius=R, color=255, thickness=-1)
                    
                    
        samplepoint_map = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for node in graph.keys():
            node_y, node_x = node
            cv.circle(samplepoint_map, (node_x, node_y), radius=R, color=255, thickness=-1)
            
        road_map = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for node, neis in graph.items():
            node_y, node_x = node
            for nei in neis:
                nei_y, nei_x = nei
                cv.line(road_map, (node_x, node_y), (nei_x, nei_y), color=255, thickness=ROAD_WIDTH)
        
        keypoint_save_path = os.path.join(ROOT_DIR, "keypoint_mask", f'{i}.png')
        samplepoint_save_path = os.path.join(ROOT_DIR, "samplepoint_mask", f'{i}.png')
        road_save_path = os.path.join(ROOT_DIR, "road_mask", f'{i}.png')
        check_dir(os.path.dirname(keypoint_save_path))
        check_dir(os.path.dirname(samplepoint_save_path))
        check_dir(os.path.dirname(road_save_path))
        
        cv.imwrite(keypoint_save_path, keypoint_map)
        cv.imwrite(samplepoint_save_path, samplepoint_map)
        cv.imwrite(road_save_path, road_map)
        
if "__main__" == __name__:
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cityscale')
    args = parser.parse_args()

    if args.dataset == 'spacenet':
        ROOT_DIR = './data/spacenet'
        IMAGE_SIZE = 400
        GRAPH_PATTERN = './data/spacenet/RGB_2.0_meter/{}__gt_graph_dense.p'
    elif args.dataset == 'cityscale':
        ROOT_DIR = './data/cityscale'
        IMAGE_SIZE = 2048
        GRAPH_PATTERN = './data/cityscale/20cities/region_{}_graph_gt.pickle'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    main()
    
