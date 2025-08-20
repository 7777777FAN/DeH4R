import numpy as np
import math
import sys
import pickle
import graph as splfy
import topo as topo
import json
#import TOPORender
import os

#  改为多线程计算
from multiprocessing import Pool
from functools import partial

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-graph_gt', default='',action='store', dest='graph_gt', type=str,
                    help='ground truth graph (in xy coordinate)')

parser.add_argument('-graph_prop', default='',action='store', dest='graph_prop', type=str,
                    help='proposed graph (in xy coordinate)')

parser.add_argument('-output', default='', action='store', dest='output', type=str,
                    help="outputfile with '.txt' as suffix")                  

parser.add_argument('-matching_threshold', action='store', dest='matching_threshold', type=float,
                    help='topo marble-hole matching distance ', required =False, default=0.00010)

parser.add_argument('-interval', action='store', dest='topo_interval', type=float,
                    help='topo marble-hole interval ', required =False, default=0.00005)

parser.add_argument('-savedir', type=str)
parser.add_argument('-trace', type=lambda x: x.lower() == "true", default=False)

args = parser.parse_args()
print(args)


lat_top_left = 41.0 
lon_top_left = -71.0 
min_lat = 41.0 
max_lon = -71.0 


def worker(tile_idx, args):

    suffix = 'with' if args.trace else 'no'
    graph_prop = '../%s/decode_result/%s_graph_%s_tracing.p'%(args.savedir,tile_idx, suffix)
    graph_gt = '../data/spacenet/RGB_1.0_meter/%s__gt_graph_dense.p'%tile_idx
    output_path = '../%s/results/topo/%s.txt'%(args.savedir,tile_idx)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    map1 = pickle.load(open(graph_gt, "rb"))
    map2 = pickle.load(open(graph_prop, "rb"))


    def xy2latlon(x,y):
        lat = lat_top_left - x * 1.0 / 111111.0
        lon = lon_top_left + (y * 1.0 / 111111.0) / math.cos(math.radians(lat_top_left))

        return lat, lon 


    def create_graph(m):
        hashable = False
        if not hashable:    # 防止点的坐标不可哈希
            new_m = {}
            for k, v in m.items():
                new_v = []
                for nei in v:
                    new_v.append(tuple(nei))
                new_m[k] = new_v
            m = new_m
            
        global min_lat 
        global max_lon 

        graph = splfy.RoadGraph() 

        nid = 0 
        idmap = {}

        def getid(k, idmap):
            
            if k in idmap :
                return idmap[k]
        
            idmap[k] = nid 
            nid += 1 

            return idmap[k]


        for k, v in m.items():
            n1 = k 

            lat1, lon1 = xy2latlon(n1[0],n1[1])

            if lat1 < min_lat:
                min_lat = lat1 

            if lon1 > max_lon :
                max_lon = lon1 

            for n2 in v:
                lat2, lon2 = xy2latlon(n2[0],n2[1])

                if n1 in idmap:
                    id1 = idmap[n1]
                else:
                    id1 = nid 
                    idmap[n1] = nid 
                    nid = nid + 1

                if n2 in idmap:
                    id2 = idmap[n2]
                else:
                    id2 = nid 
                    idmap[n2] = nid 
                    nid = nid + 1

                graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)
        
        graph.ReverseDirectionLink() 

        for node in graph.nodes.keys():
            graph.nodeScore[node] = 100

        for edge in graph.edges.keys():
            graph.edgeScore[edge] = 100


        return graph 


    graph_gt = create_graph(map1)
    graph_prop = create_graph(map2)

    print("load gt/prop graphs")

    region = [min_lat-300 * 1.0/111111.0, lon_top_left-500 * 1.0/111111.0, lat_top_left+300 * 1.0/111111.0, max_lon+500 * 1.0/111111.0]

    graph_gt.region = region
    graph_prop.region = region

    #pickle.dump(RoadGraph, open(sys.argv[8].replace('txt','graph'),"w"))
    #TOPORender.RenderGraphSVG(graph_gt, graph_prop, sys.argv[3].replace('txt','svg'))

    losm = topo.TOPOGenerateStartingPoints(graph_gt, region=region, image="NULL", check = False, direction = False, metaData = None)

    lmap = topo.TOPOGeneratePairs(graph_prop, graph_gt, losm, threshold = 0.00010, region=region)

    # for spacenet, use a smaller distance
    r = 0.00150 # around 150 meters

    topoResult =  topo.TOPOWithPairs(graph_prop, graph_gt, lmap, losm, r =r, step = args.topo_interval, threshold = args.matching_threshold, outputfile = output_path, one2oneMatching = True, metaData = None)

    print('=========',output_path,'==================')
    # print([losm, topoResult, region])
    # print(topoResult)
    #TOPORender.RenderGraphSVGMap(graph_gt, graph_prop, sys.argv[3].replace('txt','topo.svg'), topoResult)

    pickle.dump([losm, topoResult, region],  open(output_path.replace('txt','topo.p'),'wb'))






if '__main__' == __name__:
    args = parser.parse_args()
    print(args)
    with open('../data/spacenet/data_split.json','r') as jf:
        tile_list = json.load(jf)['test']
        worker_with_args = partial(worker, args=args)
        pool = Pool()
        pool.map(worker_with_args, tile_list)
        pool.close()
        pool.join()