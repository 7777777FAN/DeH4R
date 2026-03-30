# coding: utf-8

import rtree
import math
import numpy as np

class LabelGnerator():
    def __init__(self, cfg, graph):
        # graph: adj_dict
        self.cfg = cfg
        self.graph_tree = rtree.index.Index()
        self.graph = self._kick_off_invalid_edges(graph)
        if self.cfg.DATASET == 'spacenet':
            self.graph = self._transform_coord(self.graph)  # spacenet的坐标要上下翻转
        if self.cfg.DATASET == 'globalscale' and not isinstance(list(graph.keys())[0][0], int):   # 需要把graph转换为整数
            self.graph = self._float2int_graph(self.graph)
            
            
        self.all_points_coords = np.array([p for p in self.graph.keys()])
        for idx, node in  enumerate(self.all_points_coords):
            r, c = node
            self.graph_tree.insert(idx, (r, c, r, c))
            
    
   
    def _kick_off_invalid_edges(self, graph):
        new_graph = {}
        for k, n in graph.items():
            src_r, src_c = k
            neighbors = []
            if len(n):
                for nei in n:
                    dst_r, dst_c = nei
                    delta_r = dst_r - src_r
                    delta_c = dst_c - src_c
                    d = math.sqrt(delta_r**2 + delta_c**2)
                    if d < self.cfg.D_THR:  # 可能因此导致某些点被剔除
                        neighbors.append(nei)
                if len(neighbors):
                    new_graph[k] = neighbors
                else:
                    continue    # 跳过因为距离过滤而产生的孤立点
            else:   # 跳过原本就有的孤立点
                continue
            
        return new_graph
    
    
    def _transform_coord(self, graph):
        '''coord transform for spacenet'''
        new_graph = {}
        for k, n in graph.items():
            src_r, src_c = k
            new_neighbors = []
            for nei in n:
                dst_r, dst_c = nei
                new_nei = (self.cfg.IMAGE_SIZE - dst_r, dst_c)
                new_neighbors.append(new_nei)
            new_k = (self.cfg.IMAGE_SIZE - src_r, src_c)
            new_graph[new_k] = new_neighbors
            
        return new_graph
    
    def _float2int_graph(self, graph):
        new_graph = {}
        for k, n in graph.items():
            new_k = (int(k[0]+0.5), int(k[1]+0.5))
            new_neighbors = []
            for nei in n:
                new_nei = (int(nei[0]+0.5), int(nei[1]+0.5))
                new_neighbors.append(new_nei)
            new_graph[new_k] = new_neighbors
            
        return new_graph
        
        
                    
    
    def sample_patch(self, patch_info, rot_times=0):
        max_sample_points = self.cfg.TOPO_DECODER.NUM_POINTS
        (begin_r, begin_c), (end_r, end_c) = patch_info
        query_box = (begin_r, begin_c, end_r, end_c)

        patch_points_indices = list(set(self.graph_tree.intersection(query_box)))
        patch_points_indices = np.array(patch_points_indices)
        np.random.shuffle(patch_points_indices)
        patch_points_indices = patch_points_indices[:max_sample_points] if len(patch_points_indices) > max_sample_points else patch_points_indices
       
        all_null_flag = False
        if 0 == len(patch_points_indices):
            fake_points_coords = [[0, 0]] * max_sample_points
            fake_next_nodes_classes_and_coords = [([1], np.array([(0, 0)]))] * max_sample_points
            fake_next_nodes_classes_and_coords[0] = ([0], np.array([(0, 0)]))
            fake_valid = [False] * max_sample_points
            fake_valid[0] = True
            all_null_flag = True
            
            return fake_points_coords, fake_next_nodes_classes_and_coords, fake_valid, all_null_flag
        
        else:  
            patch_points_coords = self.all_points_coords[patch_points_indices, :] 
            # aug    
            # tanslation
            trans = np.array(
                [
                    [1, 0, -0.5 * self.cfg.PATCH_SIZE],
                    [0, 1, -0.5 * self.cfg.PATCH_SIZE],
                    [0, 0, 1],
                ],
                dtype=np.float32
            )
            # rotation
            rot_mat = np.array( # defined as in the Cartesian coordinate system and functions like rot*[x, y, 1].T
                [
                    [0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1],
                ],
                dtype=np.float32
            )
            patch_points_coords_aug = patch_points_coords - np.array([begin_r, begin_c])    # norm to patch
            patch_points_coords_aug = np.concatenate([patch_points_coords_aug, np.ones(shape=(len(patch_points_coords_aug), 1))], axis=1)   # to homo coord
            patch_points_coords_aug = patch_points_coords_aug @ trans.T @ np.linalg.matrix_power(rot_mat.T, rot_times) @ np.linalg.inv(trans.T)
            patch_points_coords_aug = patch_points_coords_aug[:, 0:2]   # from homo to rc
            # add noise
            noise = np.random.normal(loc=0, scale=1, size=patch_points_coords_aug.shape)*self.cfg.NOISE
            patch_points_coords_aug = patch_points_coords_aug + noise
            patch_points_coords_aug = np.clip(patch_points_coords_aug, a_min=0.0, a_max=self.cfg.PATCH_SIZE-1.0)   # [0, 511]
            
            assert len(patch_points_coords_aug) == len(patch_points_coords), 'aug 前后长度应相同'
            
            # 处理邻接点相对坐标
            next_nodes_classes_and_coords = [None] * len(patch_points_coords_aug)  # [[[Δr, Δc] ... [Δr, Δc]], ... [[...]]] 
            valid = [True] * len(patch_points_coords_aug)  # 当前点是否有效
            adj_coord_rot_mat = np.array(   # defined as in the Cartesian coordinate system and functions like rot*[x, y, 1].T
                [
                    [0, -1],
                    [1,  0],
                ],
                dtype=np.float32
            )
            
            for idx, pnt in enumerate(patch_points_coords): # 原graph的坐标是没有经过旋转的
                (r, c) = int(pnt[0]+0.5), int(pnt[1]+0.5)
                # 相对坐标可以不用norm to patch
                neighbors = self.graph[(r, c)] # abs  r，c coord
                # 处理邻接点超出当前patch的情况 -> 丢掉这个点（置为左上角坐标并标记为无效）
                is_outside = False
                if (r > begin_r+32) and (r < end_r-32) and  (c > begin_c+32) and (c < end_c-32):    # 在安全范围内
                    pass    # 先判断当前关键点坐标是否可能导致邻接点出界，这样可以不用对每个邻接点进行判断
                else:   
                    for nei in neighbors:
                        nei_r, nei_c = int(nei[0]), int(nei[1])
                        if (nei_r < begin_r) or (nei_r > end_r-1) or (nei_c < begin_c) or (nei_c > end_c-1):    
                            is_outside = True
                            patch_points_coords_aug[idx] = np.array([0, 0]) 
                            next_nodes_classes_and_coords[idx] = ([1], np.array([(0, 0)]))
                            valid[idx] = False  
                            break   
                if is_outside:  # 但凡有一个邻接点出现超出边界就丢掉当前关键点（其他patch上会正确处理的）
                    continue
                else:
                    neighbors = np.array(neighbors) - np.array([begin_r, begin_c])  # norm to patch
                    neighbors = neighbors - (self.cfg.PATCH_SIZE//2)  # trans to patch midddle
                    neighbors = neighbors @ np.linalg.matrix_power(adj_coord_rot_mat.T, rot_times) + (self.cfg.PATCH_SIZE//2) # rots and trans
                    # to relative
                    next_nodes_rel_coords = neighbors - patch_points_coords_aug[idx]     # 注意这里要减去aug后的patch坐标才正确 
                    next_nodes_rel_coords = next_nodes_rel_coords / self.cfg.NORM_D     # norm to [-1, 1]
                    next_nodes_classes_and_coords[idx] = ([0]*len(next_nodes_rel_coords), next_nodes_rel_coords)
                    valid[idx] = True   # default
            
            # 不足256个点要pad
            patch_points_coords_aug = patch_points_coords_aug.tolist()  # 便于append
            for _ in range(len(patch_points_coords_aug), max_sample_points):
                patch_points_coords_aug.append([0, 0])
                next_nodes_classes_and_coords.append(([1], np.array([(0, 0)])))     # pad的点都是1类， i.e. 非邻接点
                valid.append(False)
            
            return patch_points_coords_aug, next_nodes_classes_and_coords, valid, all_null_flag
        