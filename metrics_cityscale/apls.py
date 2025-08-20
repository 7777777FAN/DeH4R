import numpy as np
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)

args = parser.parse_args()

apls = []
name_list = [int(x.split('.')[0]) for x in os.listdir(f'../{args.dir}/results/apls')]
name_list.sort()
for file_name in name_list:
    try:
        with open(f'../{args.dir}/results/apls/{file_name}.txt') as f:
            lines = f.readlines()
        print(file_name,lines[0].split(' ')[-1][:-2])
        apls.append(float(lines[0].split(' ')[-1][:-2]))
    except:
        break
    
print('APLS',np.mean(apls))

score_dir = os.path.join(f'../{args.dir}', 'score')
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
    
with open(f'../{args.dir}/score/apls.json','w') as jf:
    json.dump({'apls':apls,'final_APLS':np.mean(apls)}, jf)