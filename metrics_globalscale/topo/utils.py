# 图数据中包含负值，需要过滤
from rtree import index
from collections import defaultdict
import numpy as np

# graph裁剪
def within_rect(point, rect):
    '''
    rect: 由4个边界值定义, (start_r, start_c, end_r, end_c)
    也即 (up, down, left, right)
    '''
    up, left, down, right = rect
    r, c = point
    if (r < up) or \
        (r > down-1) or \
        (c < left) or \
        (c > right-1):
            return False
    return True


def at_bounds(point, rect):
    if within_rect(point, rect):
        r, c = point
        up, left, down, right = rect
        if (abs(r-up) < 1e-6) or \
            (abs(r-(down-1)) < 1e-6) or \
            (abs(c-left) < 1e-6) or \
            (abs(c-(right-1)) < 1e-6):
                return True
    return False


def get_pnt_at_bound(n, nei, rect):
    in_n, out_n = n, nei
    while True:
        cal_mid = ((in_n[0]+out_n[0])/2.0, (in_n[1]+out_n[1])/2.0)	# 实际计算以浮点进行
        # real_mid = [int(x+0.5) for x in cal_mid] # int
        real_mid = cal_mid
        if at_bounds(real_mid, rect):
            break
        if within_rect(real_mid, rect):
            in_n, out_n = cal_mid, out_n
        else:
            in_n, out_n = in_n, cal_mid
    return tuple(real_mid)
   

def crop_graph(
    graph: dict, up, left, crop_size=512, expand=32
):
    
    '''
    graph: 邻接字典，坐标系为rc坐标 (必须为整数)
    up, left 分别为裁剪区域的上边界坐标和左边界坐标
    '''
    nodes = list(graph.keys())
    # 确定图的坐标的数据类型
    if isinstance(nodes[0][0], int):
        is_int = True
    elif isinstance(nodes[0][0], float):
        is_int = False
    else:
        raise TypeError(f"Unsupported type {type(nodes[0][0])} for node coordinates.")
        
    nodes = np.array(nodes)
    down, right = up+crop_size, left+crop_size
    actual_rect= (up, left, down, right)
    # 外扩以确定查找区域
    node_rtree = index.Index()
    for i, node in enumerate(nodes):
        r, c = node
        node_rtree.insert(i, (r, c, r, c))
    search_rect = tuple([up-expand, left-expand, down+expand, right+expand])
    
    nodes_idxs = list(node_rtree.intersection(search_rect))
    candidate_nodes = nodes[nodes_idxs]
    
    temp_graph = defaultdict(list)
    for n in candidate_nodes: 
        n = tuple(n)    # make hashable
        if not within_rect(n, actual_rect):
            continue
        neis = graph[n]
        new_neis = []
        for nei in neis:
            new_nei = nei
            if not within_rect(nei, actual_rect):	# 找出与裁剪边界的交点
                new_nei = get_pnt_at_bound(n, nei, actual_rect)
            new_neis.append(tuple(new_nei))
            if tuple(new_nei) not in temp_graph:
                temp_graph[tuple(new_nei)] = [n]
            elif tuple(n) not in temp_graph[tuple(new_nei)]: 
                temp_graph[tuple(new_nei)].append(n)
        temp_graph[n] = new_neis
    
    # 转换为当前patch的相对坐标
    new_graph = defaultdict(list)
    start = (up, left)
    for n, neis in temp_graph.items():
        if is_int:
            new_k = (int(n[0]-start[0]), int(n[1]-start[1]))
            new_v = (np.array(neis) - np.array(start)).astype(int).tolist()
            new_v = [tuple(x) for x in new_v]
        else:
            new_k = (n[0]-start[0], n[1]-start[1]) # hashabe
            new_v = (np.array(neis) - np.array(start)).astype(float).tolist()
            new_v = [tuple(x) for x in new_v]
        new_graph[new_k] = new_v
    return new_graph