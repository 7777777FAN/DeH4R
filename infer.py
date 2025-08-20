
import torch
from model import R2RC
from argparse import ArgumentParser
import time
from tqdm import tqdm
import os
import math
from rtree import index
import numpy as np
from collections import defaultdict
import cv2 as cv
import pickle
import utils 



parser = ArgumentParser()
parser.add_argument(
    '--config', default='./config/R2RC.yml'
)
parser.add_argument(
    '--ckpt', default=None
)
parser.add_argument(
    '--device', default='cuda'
)
parser.add_argument(
    '--output_dir', default=None
)
parser.add_argument(
    '--dev_run', default=False, action='store_true')

parser.add_argument(
    '--OOD', default='', type=str)




def collate_infer_batch_data(batch_topo_data):
    '''
        把测试时batch中每个样本的keypoints的点数一致化，以点数最多的为准，不够的padding
    '''
    collated_data = {}
    n_points = max(len(patch_kpts) for patch_kpts in batch_topo_data['keypoints'])
    for k, v in batch_topo_data.items():
        # v: [B, num_points(, 2)]
        new_v = []
        for single_v in v:
            pad_length = n_points - single_v.shape[0]
            new_single_v = (single_v).tolist()
            for _ in range(pad_length):
                if k == 'keypoints':
                    new_single_v.append(np.array([0, 0], dtype=single_v.dtype))
                elif k == 'valid':
                    new_single_v.append(np.array(False))
            new_v.append(new_single_v)
        new_v = np.stack(new_v, axis=0) # B, N_points(, 2)
        collated_data[k] = torch.tensor(new_v)
        
    return collated_data


# @utils.cal_runtime
def infer_topo_patch_by_patch(model, all_patch_info, batch_size, all_backbone_out, keypoints):
    if not len(keypoints):
        return None
    keypoints_rtree = index.Index()
    for i, pnt in enumerate(keypoints):
        r, c = pnt
        keypoints_rtree.insert(i, (r, c , r, c))
        
    GTE = defaultdict(list)
    
    num_patch = len(all_patch_info)
    num_batch = math.ceil(num_patch // batch_size)
    for B in range(num_batch):
        batch_start_idx = B * batch_size
        batched_patch_info = all_patch_info[batch_start_idx:batch_start_idx+batch_size]
        batch_backbone_out = all_backbone_out[B]    # BCHW
        
        idx_map = []
        batch_topo_data = {
            'keypoints': [],
            'valid': [],
        }
        for patch_info in batched_patch_info:
            _, (start_r, start_c), (end_r, end_c) = patch_info
            patch_query_box = (start_r, start_c, end_r, end_c)
            patch_keypoints_idx = list(keypoints_rtree.intersection(patch_query_box))
            idx_patch2all = {patch_idx:all_idx for patch_idx, all_idx in enumerate(patch_keypoints_idx)}
            idx_map.append(idx_patch2all)
            # B中的每个样本（patch）还具有不同的keypoints数目，后续还需处理
            patch_keypoints = keypoints[patch_keypoints_idx, :] - np.array([start_r, start_c], dtype=keypoints.dtype)  # 转换为相对坐标
            patch_valid = np.ones((len(patch_keypoints), ), dtype=np.bool_)
            batch_topo_data['keypoints'].append(patch_keypoints)
            batch_topo_data['valid'].append(patch_valid)
            
        batch_topo_data = collate_infer_batch_data(batch_topo_data)
        max_valid_length = batch_topo_data['keypoints'].shape[1]
        if max_valid_length == 0:
            continue
        
        # pred_adj_points_validity:  [B, max_valid_length, num_queries] 
        # pred_adj_pnt_coords: [B, max_valid_length, num_queries, 2]
        pred_adj_points_validity, pred_adj_pnt_coords = model.infer_topo(batch_backbone_out, batch_topo_data)
        pred_adj_points_validity, pred_adj_pnt_coords = pred_adj_points_validity.detach().cpu().numpy(), pred_adj_pnt_coords.detach().cpu().numpy()
        
        for b in range(batch_size):
            for n in range(max_valid_length):
                if not batch_topo_data['valid'][b, n]:  # 当前点是否是有效输入点，输入要先有效才能考虑输出
                    continue
                kpt_idx_in_all = idx_map[b][n]
                GTE[kpt_idx_in_all].append((pred_adj_points_validity[b, n], pred_adj_pnt_coords[b, n]))
        # GTE = {
        #    kpt_idx1: [ ([cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), [(cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), ... ]
        #    kpt_idx2: [ ([cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), [(cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), ... ]
        #    ...
        #    kpt_idxn: [ ([cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), [(cls1, cls2, ..., cls10], [[Δr1, Δc1], [Δr2, Δc2], ..., [Δr10, Δc10]]), ... ]
        # }
        
    return GTE
            

# @utils.cal_runtime
def infer_masks_patch_by_patch_and_vis(cfg, num_batch, batch_size, all_patch_info):
    image_size = cfg.IMAGE_SIZE
    # get keypoints
    fused_keypoint_map = torch.zeros((image_size, image_size), dtype=torch.float32, requires_grad=False).to(device)
    fused_samplepoint_map = torch.zeros((image_size, image_size), dtype=torch.float32, requires_grad=False).to(device)
    fused_road_map = torch.zeros((image_size, image_size), dtype=torch.float32, requires_grad=False).to(device)
    pixel_counter = torch.zeros((image_size, image_size), dtype=torch.float32, requires_grad=False).to(device)
    
    all_backbone_out = []
    
    for B in range(num_batch):
        batch_start_idx = batch_size * B
        batched_patch_info = all_patch_info[batch_start_idx:batch_start_idx+batch_size]
        batched_img_patches = utils.get_batch_patch(img, batched_patch_info)
        
        batched_img_patches = batched_img_patches.to(device)
        batched_backbone_out, batched_pred_mask = model.infer_img_embeddings_and_maps(batched_img_patches)    # BCHW
        all_backbone_out.append(batched_backbone_out) # [BCHW, BCHW, ...]

        for patch_idx, patch_info in enumerate(batched_patch_info):
            _, (start_r, start_c), (end_r, end_c) = patch_info
            patch_pred_mask = batched_pred_mask[patch_idx, :, :, :]  # CHW

            fused_keypoint_map[start_r:end_r, start_c:end_c] += patch_pred_mask[0, :, :]
            fused_samplepoint_map[start_r:end_r, start_c:end_c] += patch_pred_mask[1, :, :]
            fused_road_map[start_r:end_r, start_c:end_c] += patch_pred_mask[2, :, :]
            
            pixel_counter[start_r:end_r, start_c:end_c] += (torch.ones(size=patch_pred_mask[0, :, :].shape[0:2], requires_grad=False).to(fused_road_map))
            
    # fuse and vis mask
    pixel_counter = pixel_counter.detach().cpu().numpy()
    fused_keypoint_map = fused_keypoint_map.detach().cpu().numpy()
    fused_samplepoint_map = fused_samplepoint_map.detach().cpu().numpy()
    fused_road_map = fused_road_map.detach().cpu().numpy()
    
    pixel_counter[pixel_counter==0] = 1
    fused_keypoint_map /= pixel_counter
    fused_samplepoint_map /= pixel_counter
    fused_road_map /= pixel_counter
    
    mask_output_dir = os.path.join(output_dir, 'pred_mask')
    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)
    cv.imwrite(os.path.join(mask_output_dir, f'{img_id}_pred_keypoints.png'), (fused_keypoint_map*255).astype(np.uint8))
    cv.imwrite(os.path.join(mask_output_dir, f'{img_id}_pred_samplepoints.png'), (fused_samplepoint_map*255).astype(np.uint8))
    cv.imwrite(os.path.join(mask_output_dir, f'{img_id}_pred_road.png'), (fused_road_map*255).astype(np.uint8))
    
    # ======== topo ========
    fused_keypoint_map = fused_keypoint_map / max(np.amax(fused_keypoint_map), 0.001)
    fused_samplepoint_map = fused_samplepoint_map / max(np.amax(fused_samplepoint_map), 0.001)
    fused_road_map = fused_road_map / max(np.amax(fused_road_map), 0.001)
    
    keypoints, keypoint_scores = utils.detect_local_minima(arr=-fused_keypoint_map, mask=fused_keypoint_map, threshold=config.INFER.KPT_THR)
    samplepoints, samplepoints_scores = utils.detect_local_minima(arr=-fused_samplepoint_map, mask=fused_samplepoint_map, threshold=config.INFER.SMPT_THR)
    
    # 直接从路面结果NMS得到路面点
    road_points, road_scores = utils.get_points_and_scores_from_mask(mask=fused_road_map, threshold=config.INFER.ROAD_THR, mode='normal')  # 所有可能的路面点，不是细化过后的
    
    keypoints = utils.nms_points(keypoints, keypoint_scores, radius=cfg.INFER.KPT_NMS_RADIUS, return_indices=False)
    samplepoints = utils.nms_points(samplepoints, samplepoints_scores, radius=cfg.INFER.SMPT_NMS_RADIUS, return_indices=False)
    
    kpts = np.concatenate([keypoints, samplepoints, road_points], axis=0)   
    kpt_scores = np.concatenate([np.ones(keypoints.shape[0]) + 0.1, np.ones(samplepoints.shape[0]) + 0.1, road_scores], axis=0)  # 保留采样点，路面点作为补充
    
    keypoints = utils.nms_points(kpts, kpt_scores, radius=cfg.INFER.ROAD_NMS_RADIUS, return_indices=False)
    
    # vis all keypoints
    keypoints_map = np.zeros_like(fused_samplepoint_map)
    for (r, c) in keypoints:
        cv.circle(keypoints_map, (c, r), radius=3, color=(255, 255, 255), thickness=-1)
    cv.imwrite(os.path.join(mask_output_dir, f'{img_id}_keypoints.png'), keypoints_map)
    
    # vis keypoints on a image
    bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    for (r, c) in keypoints:
        cv.circle(bgr, (c, r), radius=4, color=(0, 255, 255), thickness=-1)
    cv.imwrite(os.path.join(mask_output_dir, f'{img_id}_keypoints_on_image.png'), bgr)
    
    return fused_samplepoint_map, all_backbone_out, keypoints
    

@torch.inference_mode()
def infer_one_img(cfg, model, img_id, img):
    '''
    :param img (numpy.ndarray): HWC shaped
    '''
    image_size = cfg.IMAGE_SIZE
    batch_size = cfg.INFER.INFER_BATCH_SIZE
    assert image_size == img.shape[0]
    
    all_patch_info = utils.get_patch_info_one_img(
        0, cfg.IMAGE_SIZE, cfg.PATCH_SIZE, cfg.INFER.SAMPLE_MARGIN, cfg.INFER.PATCHES_PER_EDGE
    )
    num_patch = len(all_patch_info)
    num_batch = math.ceil(num_patch // batch_size)
    
    fused_samplepoint_map, all_backbone_out, keypoints = infer_masks_patch_by_patch_and_vis(
        cfg=cfg,
        num_batch=num_batch,
        batch_size=batch_size,
        all_patch_info=all_patch_info
    )
    
    GTE = infer_topo_patch_by_patch(    # 这里是初步预测，还没在patch间进行merge
        model=model,
        all_patch_info=all_patch_info,
        batch_size=batch_size,
        all_backbone_out=all_backbone_out,
        keypoints=keypoints
    )
    
    utils.vis_GTE(
        cfg=cfg, 
        GTE=GTE, 
        keypoints=keypoints, 
        img_id=img_id, 
        img=img, 
        output_dir=output_dir
    )
    # (DO modify the GTE) merge across patches to make every kpt has unique predict for one adj point
    rel_GTE = utils.merge_across_patch_predicts(cfg, GTE, keypoints)    
    if cfg.INFER.DECODE:    # 是否使用解码算法
        total_refine = False if cfg.INFER.TRACE else True
        raw_rel_GTE, raw_decode_graph, refined_graph = utils.GTE_decode(
            cfg, 
            output_dir=output_dir,
            img_id=img_id, 
            keypoints=keypoints, 
            rel_GTE=rel_GTE, 
            snap=True, 
            snap_dist=cfg.INFER.SNAP_DIST, 
            angle_distance_weight=cfg.INFER.ANGLE_DISTANCE_WEIGHT,       
            total_refine=total_refine
        )
        graph_without_tracing = refined_graph 
    else:
        raw_decode_graph = None
        refined_graph = None
        graph_without_tracing = None
        raw_rel_GTE = rel_GTE
    
    if cfg.INFER.TRACE:
        if cfg.INFER.DECODE:
            assert graph_without_tracing is not None
            adj_dict = graph_without_tracing
            starting_points = None
            mode = 'after_decode'
            # raw_decode_graph = raw_decode_graph
        else:
            adj_dict = None
            starting_points = keypoints
            mode = 'direct'
            # raw_decode_graph = None
        
        graph_with_tracing, extended_graph_part = utils.graph_growing(
            model=model,
            cfg=cfg, 
            raw_rel_GTE=raw_rel_GTE,
            raw_keypoints=keypoints,    # 分割提取出来的原始关键点
            # raw_decode_graph=raw_decode_graph,      # 由分割结果经过解码获得的初步图（有的keypoints被过滤掉了，但存在大量的环路，需要deloop）
            adj_dict=adj_dict,      # 在decode中经过初步refine的图
            starting_points=starting_points,
            mode=mode,
            all_backbone_out=all_backbone_out,
            all_patch_info=all_patch_info,
            total_refine=True
        )  
    else:
        graph_with_tracing, extended_graph_part = None, None
        
    return fused_samplepoint_map, graph_without_tracing, graph_with_tracing, extended_graph_part


def vis_pred_graph(cfg, vis_dir, img_id, rgb, graph_without_tracing, graph_with_tracing=None, extended_graph_part=None):
    
    rgb = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    
    if cfg.INFER.DECODE and not cfg.INFER.TRACE:    # 是否使用解码算法
        assert graph_without_tracing is not None  # decode 结果
        for n, v in graph_without_tracing.items():
            src_y, src_x = n
            for nei in v:
                dst_y, dst_x = nei
                cv.line(rgb, (src_x, src_y), (dst_x, dst_y), color=(15, 160, 253), thickness=4)
                cv.circle(rgb, (dst_x, dst_y), radius=4, color=(0, 255, 255), thickness=-1)
            cv.circle(rgb, (src_x, src_y), radius=4, color=(0, 255, 255), thickness=-1) 
        output_file = os.path.join(vis_dir, f"{img_id}_pred_graph(decode).png")   
    
    if cfg.INFER.DECODE and cfg.INFER.TRACE:
        assert graph_with_tracing is not None and extended_graph_part is not None
        for n, v in graph_with_tracing.items():
            src_y, src_x = n
            for nei in v:
                dst_y, dst_x = nei
                cv.line(rgb, (src_x, src_y), (dst_x, dst_y), color=(15, 160, 253), thickness=4)
                cv.circle(rgb, (dst_x, dst_y), radius=4, color=(0, 255, 255), thickness=-1)
            cv.circle(rgb, (src_x, src_y), radius=4, color=(0, 255, 255), thickness=-1)
  
        for n, v in extended_graph_part.items():
                src_y, src_x = n
                if n not in graph_with_tracing:
                    continue
                for nei in v:
                    dst_y, dst_x = nei
                    if nei not in graph_with_tracing[n]:
                        continue
                    cv.line(rgb, (src_x, src_y), (dst_x, dst_y), color=(255, 255, 0), thickness=4)
                    cv.circle(rgb, (dst_x, dst_y), radius=4, color=(0, 0, 255), thickness=-1)
                cv.circle(rgb, (src_x, src_y), radius=4, color=(0, 0, 255), thickness=-1) 
        output_file = os.path.join(vis_dir, f"{img_id}_pred_graph(decode+trace).png") 
        
    if not cfg.INFER.DECODE and cfg.INFER.TRACE:
        assert graph_with_tracing is not None
        for n, v in graph_with_tracing.items():
            src_y, src_x = n
            for nei in v:
                dst_y, dst_x = nei
                cv.line(rgb, (src_x, src_y), (dst_x, dst_y), color=(15, 160, 253), thickness=4)
                cv.circle(rgb, (dst_x, dst_y), radius=4, color=(0, 255, 255), thickness=-1)
            cv.circle(rgb, (src_x, src_y), radius=4, color=(0, 255, 255), thickness=-1)
        output_file = os.path.join(vis_dir, f"{img_id}_pred_graph(trace).png")
        
    cv.imwrite(output_file, rgb)
    


if '__main__' == __name__:
    args = parser.parse_args()
    config = utils.load_config(args.config)
    config.dev_run = False
    device = torch.device(args.device) if 'cuda' in args.device else torch.device('cpu')
    
    torch.backends.cudnn.enabled = True # 启用cuDNN
    torch.backends.cudnn.benchmark = True # 为当前设置寻找合适的算法
    # utils.set_seed()
    
    # init model and load ckpts
    model = R2RC(config)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    print(f"=========== loading ckpt from {args.ckpt} to model ===========")
    model.load_state_dict(ckpt['state_dict'], strict=True)
    print(f"=========== {'ckpt successfully loaded!':} ===========")
    model.to(device)
    model.eval()
    # get data partion
   
    if str(config.DATASET).lower() == 'cityscale':
        rgb_pattern = './data/cityscale/20cities/region_{}_sat.png'
        _, _, test_img_ids = utils.get_data_split(dataset=config.DATASET)
        
    elif str(config.DATASET).lower() == 'spacenet':
        rgb_pattern = './data/spacenet/RGB_1.0_meter/{}__rgb.png'
        _, _, test_img_ids = utils.get_data_split(dataset=config.DATASET)
        
    elif str(config.DATASET).lower() == 'globalscale':
        rgb_pattern = './data/globalscale/all/region_{}_sat.png'
        _, _, test, test_ood = utils.globalscale_data_partition()
        if 'ood' == args.OOD.lower():
            test = test_ood
            rgb_pattern = './data/global_scale_out_of_domain/all/region_{}_sat.png'
        test_img_ids = test
    
    # get output dir
    prefix = './infer/'
    output_dir = utils.get_output_dir_and_save_config(config, prefix=prefix, sepecified_dir=f'./infer/{args.output_dir}' if args.output_dir else None)
    
    # infer every img
    time_used = 0
    for img_id in tqdm(test_img_ids):
        img = utils.read_rgb(rgb_pattern.format(img_id))
        start_time = time.time()
        
        # GTE decode的graph， 附加追踪结果的graph，单独的追踪部分graph
        assert config.INFER.DECODE or config.INFER.TRACE, "Use at least one algorithm"
        pred_samplepoints_mask, pred_graph_without_tracing, pred_graph_with_tracing, extended_graph_part = infer_one_img(config, model, img_id, img)
        
        end_time = time.time()
        time_used += (end_time - start_time)
        
        # vis graph with rgb as base map
        vis_dir = os.path.join(output_dir, 'graph_vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            
        vis_pred_graph(cfg=config, 
                       vis_dir=vis_dir, 
                       img_id=img_id, 
                       rgb=img, 
                       graph_without_tracing=pred_graph_without_tracing, 
                       graph_with_tracing=pred_graph_with_tracing, 
                       extended_graph_part=extended_graph_part
        )
        # save graph
        GTE_decode_output_dir = os.path.join(output_dir, 'decode_result')   # output_dir/decode_result/
        if not os.path.exists(GTE_decode_output_dir):
            os.makedirs(GTE_decode_output_dir)
        filename_prefix = os.path.join(GTE_decode_output_dir, f'{img_id}')
        
        if config.INFER.TRACE:  # output pred_graph_with_tracing only when TRACE is enabled
            if str(config.DATASET).lower() == 'spacenet':   # spacenet的graph坐标需要上下翻转 
                pred_graph_with_tracing = utils.transform_coord(config, pred_graph_with_tracing)
            pickle.dump(pred_graph_with_tracing, open(filename_prefix+'_graph_with_tracing.p', 'wb'))

    print('Inference done!')
    time_txt = f'Inference finish in {time_used}s.'
    print(time_txt)
    
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w') as f:
        f.write(time_txt)
    
    
