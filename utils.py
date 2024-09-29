# coding: utf-8

# 各种工具函数
import yaml
from addict import Dict



def load_config(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return Dict(cfg)


# TODO