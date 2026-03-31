
<div align="center">
    <h1>DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction</h1>
    <a href="https://arxiv.org/abs/2508.13669" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&labelColor=555555" />
    </a>
</div>



## **Intro**
DeH4R unifies graph-growing dynamics with graph-generating efficiency through a decoupling strategy, effectively harnessing their complementary strengths, which offers great flexibility and is able to grow the graph in parallel from multiple points. DeH4R achieves new SOTA results with a significant improvement over previous methods and exceptional inference speed on two mainstream public benchmarks.

## **Updates**
- ✅ **2026/3/30**: Full code release.
- ✅ **2026/3/30**: Accepted to IEEE Transactions on Geoscience and Remote Sensing (TGRS)
- ✅ **2025/9/14**: Release model and weights (available at [DeH4R](https://huggingface.co/godx7/DeH4R)). 
- ✅ **2025/8/20**: Release the code for inference and evauation.

## **Environment setup**

To quickly set up the environment, run 
```bash
bash ./scripts/run_uv_init.sh

source .DeH4R/bin/activate
```
## **Data & Pretrained Models**
### Directory architecture
```
DeH4R
├── data/
│    ├── cityscale/
│    └── spacenet/
└── sam_ckpts/
│    ├── sam_vit_b_01ec64.pth
│    └── sam2.1_hiera_base_plus.pt
└── pretrained_pth/
     ├── cityscale_sam.pth
     ├── cityscale_sam2.pth
     ├── spacenet_sam.pth
     └── spacenet_sam2.pth
```

### Prepare data
To download data, run 
```bash
bash ./scripts/run_download_data.sh
```
or manually download the [CityScale](https://drive.google.com/file/d/1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H/view?usp=share_link) dataset into `./data/cityscale/` and then unzip it ([SpaceNet](https://drive.google.com/file/d/1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W/view?usp=share_link) dataset similarly). 🚚 Links are copied from [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus).

To preprocess data (generate 3 types of masks), run 
```bash
python data_preprocess.py --dataset dataset_name
```


### Prepare pretrained checkpoints
Download the ViT-B checkpoint from official [SAM ](https://github.com/facebookresearch/segment-anything) repo and the hiera_B+ checkpoint from [SAM2](https://github.com/facebookresearch/sam2) repo, respectively. Put them under `./sam_ckpts/`.

We provide DeH4R checkpoints for convenient inference (available at [DeH4R](https://huggingface.co/godx7/DeH4R)). You can download it and put it under `./pretrained_pth/`

## **Train**
Specify config file in `./scripts/run_train_cityscale.sh` and then run 
```bash
bash ./scripts/run_train_cityscale.sh  # SpaceNet similarly
```

## **Inference**
Specify config file and the checkpoint path in `./scripts/run_test_cityscale.sh` and then run 
```bash
bash ./scripts/run_test_cityscale.sh  # SpaceNet similarly
```
or directly run 
```bash
python infer.py --config=./config/cityscale/final_sam2.yml \
    --ckpt=./path/to/checkpoint.ckpt    # or *.pth
```
⚠️ Note: Here we specify the config file because DeH4R supports three graph building mode: `DECODE`, `TRACE` and `DECODE & TRACE`, where `TRACE` means graph expansion (growing), and we decide the mode from config.

## **Evaluation**
Specify the config file path in `run_metrics_cityscale.sh` and then run
```bash
bash ./scripts/run_metrics_cityscale.sh  /path/to/output/dir/     # SpaceNet similarly
```
⚠️ Note: The content of the config file should be the same as in inference.


## **Acknowledgement**
We would like to acknowledge that DeH4R has benefited from:
- [SAM-Road](https://github.com/htcr/sam_road)
- [Sat2Graph](https://github.com/songtaohe/Sat2Graph)
- [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus)
- [DETR](https://github.com/facebookresearch/detr)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [SAM2](https://github.com/facebookresearch/sam2)

We are grateful to their authors for making these resources available.


## **Contact**
For any questions, please email [godx](mailto:gooodx@whu.edu.cn) or, preferably, open an issue.


## **Citation**
```
@misc{gong2025deh4rdecoupledhybridmethod,
    title={DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction}, 
    author={Dengxian Gong and Shunping Ji},
    year={2025},
    eprint={2508.13669},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2508.13669}, 
}
```