
# **DeH4R**
## The official repo of [DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction.](https://arxiv.org/abs/2508.13669)

## **Updates**
- [ ] **Incoming**: Training code.
- ✅ **2025/8/20**: Release the code for inference and evauation.

## **Enviroment & Platform**
- RTX 3090 * 8
- Ubuntu 24.04.1 LTS
- CUDA: 12.0
- python 3.11.9
- torch 2.4.1
- pytorch-lightning 2.4.0
- wandb 0.18.7
- Rtree 1.3.0

Actually, traning DeH4R do not require 8 GPUs, 4 is recommended.

## **Data**
### Directory architecture
```
DeH4R
├── data/
│   ├── cityscale/
│   └── spacenet/
└──sam_ckpts/
    ├── sam_vit_b_01ec64.pth
    └── sam2.1_hiera_base_plus.pt
```
### Prepare data
To download data, run 
```bash
bash ./download_data.bash
```
or manually download the [CityScale](https://drive.google.com/file/d/1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H/view?usp=share_link) dataset into `./data/cityscale/` and then unzip it ([SpaceNet](https://drive.google.com/file/d/1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W/view?usp=share_link) dataset similarly). 

🚚 The data was neither collected, published, nor stored by us, we only provide a convenient way to download it. Links are copied from [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus).

To preprocess data (generate 3 types of masks), run 
```bash
python data_preprocess.py --dataset dataset_name
```
and then check the generated masks in corresponding dataset directory.


### Prepare pretrained checkpoints
Download the ViT-B checkpoint from official [SAM ](https://github.com/facebookresearch/segment-anything) repo and the hiera_B+ checkpoint from [SAM2](https://github.com/facebookresearch/sam2) repo, respectively. Put them under `./sam_ckpts/`

## **Train**
Incoming...

## **Inference**
### Infer on the CityScale dataset
Specify config file and the checkpoint path in `run_test_cityscale.bash` and then run 
```bash
bash ./run_test_cityscale.bash
```
or directly run 
```bash
python infer.py --config=./config/cityscale/sam.yml \
    --ckpt=./path/to/checkpoint.ckpt
```
⚠️ Note: Here we specify the config file because DeH4R supports three graph building mode: `DECODE`, `TRACE` and `DECODE & TRACE`, where `TRACE` means graph expansion (growing), and we decide the mode from config.

### Infer on the Spacenet dataset
Similar to CityScale.

## **Evaluation**
### Evaluate on the CityScale dataset
Specify the config file path in `run_metrics_cityscale.bash` and then run
```bash
bash ./run_metrics_cityscale.bash  /path/to/output/dir/
```
⚠️ Note: The content of the config file should be the same as in inference.

### Evaluate on the Spacenet dataset
Similar to CityScale.


## **Ackonwledgement**
We would like to acknowledge that DeH4R has benefited from:
- [sam_road](https://github.com/htcr/sam_road)
- [Sat2Graph](https://github.com/songtaohe/Sat2Graph)
- [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus)
- [DETR](https://github.com/facebookresearch/detr)
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [sam2](https://github.com/facebookresearch/sam2)

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