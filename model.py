# coding: utf-8

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import wandb
import os
from torchmetrics.classification import (
    BinaryJaccardIndex,
    F1Score,
    BinaryPrecisionRecallCurve,
)

from sam.segment_anything.modeling import ImageEncoderViT as SAM_Encoder
from sam.segment_anything.modeling.common import LayerNorm2d

from sam2.sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder as SAM2_Encoder
from sam2.sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.sam2.modeling.backbones.hieradet import Hiera

from functools import partial

from pprint import pprint

from matcher import build_matcher
from criterion import Criterion


class MaskDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_feature_levels = 3 if cfg.MASK_DECODER.USE_HIGH_RES_FEAT else 1 
        in_chans = cfg.ENCODER.ENCODER_OUTPUT_DIM
        activation = nn.GELU
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2),   # -->1/8
            LayerNorm2d(in_chans // 2),
            activation(),
            nn.ConvTranspose2d(in_chans // 2, in_chans // 4, kernel_size=2, stride=2), # -->1/4
            activation(),
            nn.ConvTranspose2d(in_chans // 4, in_chans // 8, kernel_size=2, stride=2), # -->1/2
            activation(),
            nn.ConvTranspose2d(in_chans // 8, 3, kernel_size=2, stride=2),    # -->1/1
        )

        if cfg.MASK_DECODER.USE_HIGH_RES_FEAT and self.cfg.sam2_activated:
            transformer_dim = cfg.ENCODER.ENCODER_OUTPUT_DIM
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)  # 1/4
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 2, kernel_size=1, stride=1)  # 1/8  
        else:
            self.conv_s0 = self.conv_s1 = None


    def forward(self, backbone_out):
        '''
        :param x: tensor or a list of multi-scale feats.
        '''
        if self.cfg.sam2_activated:
            feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:] # [1/4 ... 1/16]
            # vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            image_embedding, high_res_embedding = feature_maps[-1], feature_maps[:-1]
            if self.cfg.MASK_DECODER.USE_HIGH_RES_FEAT:
                dc1, ln1, act1, dc2, act2, dc3, act3, dc4 = self.decoder
                high_feat_s0, high_feat_s1 = self.conv_s0(high_res_embedding[0]), self.conv_s1(high_res_embedding[1])   # 1/4, 1/8
                y = act1(ln1(dc1(image_embedding)) + high_feat_s1)
                y = act2(dc2(y) + high_feat_s0)
                y = act3(dc3(y))
                y = dc4(y)
                return y
            else:
                return self.decoder(image_embedding)
        else:
            image_embedding = backbone_out
            return self.decoder(image_embedding)  # BCHW


class TopoDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_feature_levels = 3 if cfg.TOPO_DECODER.USE_HIGH_RES_FEAT else 1 # 是否丢弃经过FPN的后的1/32的特征
        self.hidden_dim = cfg.TOPO_DECODER.HIDDEN_DIM
        self.num_heads = cfg.TOPO_DECODER.NUM_HEADS
        self.depth = cfg.TOPO_DECODER.DEPTH
        self.dim_ffn = cfg.TOPO_DECODER.DIM_FFN
        self.ROI_size = cfg.TOPO_DECODER.ROI_SIZE
        self.num_queries = cfg.TOPO_DECODER.NUM_QUERIES

        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
      
        if self.cfg.CONCAT:
            self.proj = nn.Conv2d(2*self.cfg.ENCODER.ENCODER_OUTPUT_DIM,
                                self.cfg.TOPO_DECODER.HIDDEN_DIM, kernel_size=1, stride=1)
        else:
            self.proj = nn.Conv2d(self.cfg.ENCODER.ENCODER_OUTPUT_DIM,
                                self.cfg.TOPO_DECODER.HIDDEN_DIM, kernel_size=1, stride=1)
            
        if self.cfg.TOPO_DECODER.USE_HIGH_RES_FEAT and self.cfg.sam2_activated and self.cfg.TOPO_DECODER.PROJ:
            self.output_ROI_feature = nn.Conv2d(256, 256, kernel_size=1)
            self.act = nn.GELU() 
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_ffn,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.topo_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.depth)
        self.output_proj = nn.Linear(self.hidden_dim, 4)  # 输出[p1, p2, Δx, Δy] p1, p2 表示属于0类和1类的概率, 0类才是目标类别，1为无类别或背景
        # self.mlp = nn.Sequential(nn.Linear(self.hidden_dim, 2*self.hidden_dim),
        #                          nn.GELU(),
        #                          nn.Linear(2*self.hidden_dim, self.hidden_dim),
        #                          nn.GELU(),
        #                          nn.Linear(self.hidden_dim, 4)  # 输出[p1, p2, Δx, Δy] p1, p2 表示属于0类和1类的概率, 0类才是目标类别，1为无类别或背景
        #             )
        
    
    def _get_valid_grid(self, keypoints):
        """
        由关键点的rc坐标输出对应原图上ROI大小区域的归一化坐标以及对超出原图尺寸的部分进行进行标识的mask

        :param keypoints: [B, N_points, 2]

        :return grids: [B, N_points, H, W, 2]
        :return invalids: [B, N_points, H, W, 2]
        """
        B, N_points, _ = keypoints.shape
        ROI_size = self.cfg.TOPO_DECODER.ROI_SIZE
        # assert ROI_size % 2 == 0  # even size
        assert ROI_size % 2 == 1, 'ROI_SIZE must be odd size.'  # odd size
        patch_size = self.cfg.PATCH_SIZE
        # margin = ROI_size // 2
        margin = (ROI_size - 1 ) // 2
        

        batched_grid = []
        for sample in range(B):
            sample_grid = []
            for point in keypoints[sample]:
                r, c = round(point[0].cpu().item()), round(point[1].cpu().item())
                # grid_y, grid_x = torch.meshgrid([torch.arange(r-margin, r+margin), torch.arange(c-margin, c+margin)], indexing="ij")
                grid_y, grid_x = torch.meshgrid([torch.arange(r-margin, r+margin+1), torch.arange(c-margin, c+margin+1)], indexing="ij")
                grid = torch.stack([grid_x, grid_y], dim=2).float()   # [ROI_SIZE, ROI_SIZE, 2]  2 represents the coord, float type
                sample_grid.append(grid)
            sample_grid = torch.stack(sample_grid, dim=0)   # [N_points, H, W, 2]
            batched_grid.append(sample_grid)
        batched_grid = torch.stack(batched_grid, dim=0)  # [B, N_point, H, W, 2]

        # norm to [-1, 1] for grid_sample
        batched_grid = (batched_grid / (patch_size - 1)) * 2.0 - 1.0
        batched_invalid = None

        if (margin <= r <= patch_size - margin) and (margin <= c <= patch_size - margin):
            pass
        else:
            # [B, N_points, H, W]
            batched_invalid = (
                   (batched_grid[..., 0] < -1)
                | (batched_grid[..., 1] < -1)
                | (batched_grid[..., 0] >  1)
                | (batched_grid[..., 1] >  1)
            ).bool()  # mask for invalid position
            # -> [B, N_points, H, W, 2]
            batched_invalid = batched_invalid.unsqueeze(-1).expand(-1, -1, -1, -1, 2)  # explicitly match the shape of batched_grid
            batched_grid[batched_invalid] = 0.0
        
        return batched_grid, batched_invalid


    def cal_ROI_feature(self, image_embeddings, keypoints):
        """
        :Note: 
            wholly upsample the feature to patch_size is unacceptable(num_channel is big!) \
            when it comes to memery consideration, and thus we choose to calculate on the fly \
            for specific regionrather than the full size.
        """
        # image_embeddings: [B, C, H, W]
        # keypoints: [B, N_points, 2]
        B, C, _, _ = image_embeddings.shape
        _, N_points, _ = keypoints.shape
        H, W = self.cfg.TOPO_DECODER.ROI_SIZE, self.cfg.TOPO_DECODER.ROI_SIZE
        # [B, N_points, H, W, 2]
        grid, invalid = self._get_valid_grid(keypoints)
        # reshape for grid_sample
        grid = grid.reshape(B, N_points, H*W, 2).to(image_embeddings.device)
        # [B, C, N_points, H*W]
        batch_caled_features = F.grid_sample(image_embeddings, grid, mode="bilinear", align_corners=False)
        if invalid is not None:     # there is ROI outside the original patch, need to set their value to zero along channel dim
            invalid = invalid.reshape(B, N_points, H*W, 2).to(image_embeddings.device)
            invalid = invalid[..., 0].unsqueeze(1).expand(-1, C, -1, -1) # [B, 1, N_points, H*W] -> [B, C, N_points, H*W]
            batch_caled_features[invalid] = 0.0
        # [B, C, N_points, H*W] -> [B, N_points, C, H, W]
        batch_caled_features = batch_caled_features.permute(0, 2, 1, 3).reshape(B, N_points, C, H, W)

        return batch_caled_features
    
    
    def _get_ROI_feature_form_sam(self, backbone_out, keypoints=None, upsampler=None):
        # image_embeddings: [B, 256, H, W]
        image_embeddings = backbone_out # [1/16]
        image_embeddings = backbone_out # [1/16]
        
        B, C, _, _ = image_embeddings.shape
        _, N_points, _ = keypoints.shape
        
        image_features = image_embeddings
        if self.cfg.UPSAMPLE:
            assert upsampler is not None
            image_features = upsampler(image_embeddings)    # upsampled_features: [B, 256, H, W]
        ROI_features = self.cal_ROI_feature(image_features, keypoints) # [B, N_points, C, H, W], C=256
        
        H, W = self.cfg.TOPO_DECODER.ROI_SIZE, self.cfg.TOPO_DECODER.ROI_SIZE
        if self.cfg.CONCAT:
            assert self.cfg.PATCH_SIZE // 16 == self.cfg.TOPO_DECODER.ROI_SIZE, '只有ROI_SIZE与1/16尺度的特征图相同大小时才能concat'
            sum_C = C + ROI_features.shape[2]   
            # [B, 1, C, H, W] -> [B, N_points, C, H, W] for concatenation
            image_embeddings = image_embeddings.unsqueeze(1).repeat(1, N_points, 1, 1, 1)
            # [B, N_points, C, H, W] -> [B*N_points, sum_C, H, W]
            ROI_features = torch.concat([ROI_features, image_embeddings], dim=2).reshape(-1, sum_C, H, W)
        
        return ROI_features
    
    
    def _get_ROI_feature_form_sam2(self, backbone_out, keypoints=None):
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:] # [1/4 ... 1/16]
        if self.cfg.TOPO_DECODER.WITH_POS_EMBED:
            vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
            feature_maps = [fm + pe for fm, pe in zip(feature_maps, vision_pos_embeds)]
        image_embeddings, high_res_feats = feature_maps[-1], feature_maps[:-1]
        
        ROI_features = self.cal_ROI_feature(image_embeddings=image_embeddings, keypoints=keypoints)
        if self.cfg.TOPO_DECODER.USE_HIGH_RES_FEAT:
            max_idx = len(high_res_feats) - 1 # [1/4, 1/8]
            for i in range(max_idx, -1, -1):
                # ROI_features += self.cal_ROI_feature(image_embeddings=image_embeddings, keypoints=keypoints)  # 第一次训练的要用这句来测试
                ROI_features = ROI_features + self.cal_ROI_feature(image_embeddings=high_res_feats[i], keypoints=keypoints)
            B, N_points, C, H, W = ROI_features.shape   # C=256 
            if self.cfg.TOPO_DECODER.PROJ:
                ROI_features = ROI_features.flatten(0, 1)  # [B*N_points, C, H, W]
                ROI_features = self.act(self.output_ROI_feature(ROI_features))
                ROI_features = ROI_features.view(B, N_points, -1, H, W)  
    
        return ROI_features
    
    
    def forward(self, backbone_out, keypoints=None, keypoints_valid_mask=None, upsampler=None):
        # keypoints: [B, N_points, 2]
        assert (keypoints is not None) and (keypoints_valid_mask is not None)
            
        if self.cfg.sam2_activated:
            ROI_features = self._get_ROI_feature_form_sam2(backbone_out=backbone_out, keypoints=keypoints)
        else:
            ROI_features = self._get_ROI_feature_form_sam(backbone_out=backbone_out, keypoints=keypoints, upsampler=upsampler)
            
        B, N_points, _ = keypoints.shape
        H, W = ROI_features.shape[-2:]
        ROI_features = ROI_features.reshape(B*N_points, -1, H, W)
        x = self.proj(ROI_features)  # 256->128 or 512->128
        
        _, _, H, W = x.shape    
        x = x.flatten(2).permute(0, 2, 1)    # [B*N_points,C,H,W] -> [B*N_points,H*W,C]
        
        keypoints_valid_mask = keypoints_valid_mask.reshape(-1, 1).expand(B * N_points, H*W)  # [B * N_points, 1] -> [B * N_points, H*W]
        keypoints_valid_mask[:, 0] = True    # 保证每行至少有一个位置是不被mask的以规避NAN
        keypoints_valid_mask = ~keypoints_valid_mask    # 制作样本时，如果该位置是填充的则为False，但是transformer的memory_key_padding_mask中True才对应被mask掉
        
        x = self.topo_decoder(
            tgt=self.query_embed.weight.unsqueeze(0).expand(B*N_points, -1, -1), 
            memory=x, 
            memory_key_padding_mask=keypoints_valid_mask
        )  
        
        output_logits = self.output_proj(x) # 对每个patch的每个keypoint（[1, H*W, C]）输入num_queries个256维向量并输出num_queries个4维向量
        output_logits = output_logits.reshape(B, N_points, self.num_queries, -1)

        return output_logits
            

class FeatureUpsampler(nn.Module):
    def __init__(self, cfg, in_chans=256):
        super().__init__()
        self.cfg = cfg
        activation = nn.GELU
        self.upsampler = nn.Sequential(
            # nn.ConvTranspose2d(in_chans, in_chans//2, kernel_size=2, stride=2),
            # LayerNorm2d(in_chans//2), 
            nn.ConvTranspose2d(in_chans, in_chans, kernel_size=2, stride=2),
            LayerNorm2d(in_chans), 
            activation(),
            nn.ConvTranspose2d(in_chans, in_chans, kernel_size=2, stride=2),
            LayerNorm2d(in_chans),
            activation(),
            
            # nn.ConvTranspose2d(in_chans//2//2, in_chans//2//2//2, kernel_size=2, stride=2),
            # activation(),
            # nn.ConvTranspose2d(in_chans//2//2//2, 1, kernel_size=2, stride=2)
        )


    def forward(self, image_embeddings):
        """该函数将image_embedding直接上采样到原尺寸的特征图"""
        # image_embedding: [B, C, H, W]
        # H, W = self.cfg.PATCH_SIZE, self.cfg.PATCH_SIZE
        # return F.interpolate(image_embeddings, size=(H, W), mode="bilinear", align_corners=False)
        return self.upsampler(image_embeddings) # 1/16 -> 1/8
    
    
    
class DeH4R(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # overall cfg
        self.cfg = cfg
        self.cfg.sam2_activated = 'sam2' in cfg.ENCODER.BACKBONE.lower()
        self.image_size = cfg.PATCH_SIZE

        # data cfg
        self.register_buffer("mean", torch.Tensor(cfg.MEAN).view(-1, 1, 1), persistent=False)
        self.register_buffer("std", torch.Tensor(cfg.STD).view(-1, 1, 1), persistent=False)

        assert cfg.ENCODER.BACKBONE.lower() in ["sam-vit-b", 'sam2-hiera-b+',], f"{cfg.ENCODER.BACKBONE} is not a valid backbone! "

        self.image_encoder = self._init_image_encoder(cfg)
        
        self.mask_decoder = self._init_mask_decoder()
        self.feature_upsampler = None
        if cfg.UPSAMPLE:
            self.feature_upsampler = self._init_feature_upsampler()
        self.topo_decoder = self._init_topo_decoder()

        # HungarianMatcher
        self.matcher = build_matcher(cfg.MATCHER.SET_COST_CLASS, cfg.MATCHER.SET_COST_BOX)
        # criterion
        self.criterion = Criterion(num_classes=1, matcher=self.matcher, cfg=cfg)
        # metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.samplepoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)

        
        encoder_name = '2' if self.cfg.sam2_activated else ''
        if cfg.PRETRAINED:
            print(f'Using pretrained SAM{encoder_name} Encoder weights')
            self._load_pretrained_weights()
        else:
            print(f'Training SAM{encoder_name} Encoder from scratch')


    def _init_image_encoder(self, cfg):
        if cfg.ENCODER.BACKBONE.lower() == "sam-vit-b":
            # SAM encoder cfg
            self.encoder_output_dim = cfg.ENCODER.ENCODER_OUTPUT_DIM
            self.vit_patch_size = cfg.ENCODER.VIT_PATCH_SIZE

            self.encoder_embed_dim = 768
            self.encoder_num_transformer_blocks = 12
            self.encoder_num_heads = 12
            self.encoder_global_attn_indexes = [2, 5, 8, 11]
            return SAM_Encoder(
                img_size=self.image_size,
                patch_size=self.vit_patch_size,
                in_chans=3,
                embed_dim=self.encoder_embed_dim,
                depth=self.encoder_num_transformer_blocks,
                num_heads=self.encoder_num_heads,
                mlp_ratio=4.0,
                out_chans=self.encoder_output_dim,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                use_abs_pos=True,
                use_rel_pos=True,
                window_size=14,
                global_attn_indexes=self.encoder_global_attn_indexes,
            )
        elif cfg.ENCODER.BACKBONE.lower() == "sam2-hiera-b+":
            # SAM2 encoder cfg
            self.encoder_embed_dim = 112
            self.encoder_num_heads = 2
            # pos enc.
            self.num_pos_feats = 256
            self.temperature = 10000
            # fpn neck
            self.d_model = 256
            self.backbone_channel_list = [896, 448, 224, 112]
            self.fpn_top_down_levels = [2, 3]
            self.fpn_interp_model = 'nearest'
            
            truck = Hiera(embed_dim=self.encoder_embed_dim, num_heads=self.encoder_num_heads)
            neck_pos_encoding = PositionEmbeddingSine(
                num_pos_feats=self.num_pos_feats,
                temperature=self.temperature,
                normalize=True,
                scale=None
            )
            neck = FpnNeck(
                position_encoding=neck_pos_encoding,
                d_model=self.d_model,
                backbone_channel_list=self.backbone_channel_list,
                fpn_top_down_levels=self.fpn_top_down_levels,
                fpn_interp_model=self.fpn_interp_model
            )
            return SAM2_Encoder(trunk=truck, neck=neck, scalp=1)
        
        
    def _init_mask_decoder(self):
        return MaskDecoder(self.cfg)


    def _init_topo_decoder(self):
        return TopoDecoder(self.cfg)
    
    
    def _init_feature_upsampler(self):
        return FeatureUpsampler(self.cfg)
    
    
    def _resize_sam_pos_embedding(self, pretrained_state_dict):
        """把SAM与训练权重里面的abs_pos_embed和rel_pos_embed都resize一下"""
        new_state_dict = {k: v for k, v in pretrained_state_dict.items()}
        pos_embed = new_state_dict["image_encoder.pos_embed"]  # BHWC
        token_size = int(self.image_size // self.vit_patch_size)
        if pos_embed.shape[1] != token_size:  # != 1024/16
            # abs pos
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # ->BCHW for interpolate
            pos_embed = F.interpolate(
                pos_embed,
                size=(token_size, token_size),
                mode="bilinear",
                align_corners=False,
            )
            new_state_dict["image_encoder.pos_embed"] = pos_embed.permute(0, 2, 3, 1)
            # rel_pos
            rel_pos_key_pattern = "{}.attn.rel_pos"
            global_rel_pos_keys = []
            for idx in self.encoder_global_attn_indexes:
                for k in new_state_dict.keys():
                    if rel_pos_key_pattern.format(idx) in k:
                        global_rel_pos_keys.append(k)
            for k in global_rel_pos_keys:
                rel_pos_embed = new_state_dict[k]
                H, W = rel_pos_embed.shape  # W 其实是对应的通道数
                rel_pos_embed = rel_pos_embed.unsqueeze(0).unsqueeze(0)  # HW -> BCHW [1,1,H,W]
                rel_pos_embed = F.interpolate(
                    rel_pos_embed,
                    size=(2 * token_size - 1, W),
                    mode="bilinear",
                    align_corners=False,
                )
                new_state_dict[k] = rel_pos_embed[0, 0, :, :]
                
        return new_state_dict


    def _load_pretrained_weights(self):
        with open(self.cfg.ENCODER.SAM_CKPT_PATH, "rb") as f:
            if self.cfg.ENCODER.BACKBONE.lower() == 'sam-vit-b':
                state_dict = torch.load(f, weights_only=True)
                state_dict = self._resize_sam_pos_embedding(state_dict)
            elif self.cfg.ENCODER.BACKBONE.lower() == 'sam2-hiera-b+':
                model_dict = torch.load(f, weights_only=True)
                state_dict = model_dict['model']
        
        new_state_dict = {}
        matched_names = []
        unmatched_names = []
        
        for n, p in self.named_parameters():  # name, param
            if n in state_dict and p.shape == state_dict[n].shape:
                new_state_dict[n] = state_dict[n]
                matched_names.append(n)
            else:
                unmatched_names.append(n)
        if self.cfg.dev_run:
            pprint("========== Matched names ==========")
            pprint(matched_names)
            print()
            pprint("xxxxxxxxxx Unmatched names xxxxxxxxxx")
            pprint(unmatched_names)
        self.load_state_dict(new_state_dict, strict=False)


    def forward(self, rgb, keypoints, keypoints_valid_mask):
        # rgb: [B, H, W, C]
        # keypoints: [B, N_points, 2]
        # keypoints_valid_mask: [B, N_points]
        x = rgb.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = (x - self.mean) / self.std
        
        backbone_out = self.image_encoder(x)
        mask_logits = self.mask_decoder(backbone_out).permute(0, 2, 3, 1)  # [B, 3, H, W] -> [B, H, W, 3] 
        pred_next_nodes_logits = self.topo_decoder(backbone_out, keypoints, keypoints_valid_mask, self.feature_upsampler)
        # mask_logits: [B, H, W, 3] 
        # pred_next_nodes_logits: [B, N_points, num_queries, 4]
        # [
        #   [p11, p12, Δx1, Δy1],
        #   [p21, p22, Δx2, Δy2],
        #       ...
        #   [p61, p62, Δx6, Δy6]
        # ]

        output = {}
        output["pred_mask"] = mask_logits   # B3HW
        output["pred_logits"] = pred_next_nodes_logits[..., 0:2]
        output["pred_coords"] = pred_next_nodes_logits[..., 2:4].tanh()
        # output["pred_coords"] = pred_next_nodes_logits[..., 2:4]

        return output


    def infer_img_embeddings_and_maps(self, rgb):
        x = rgb.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = (x - self.mean) / self.std
        backbone_out = self.image_encoder(x)
        mask_logits = self.mask_decoder(backbone_out)  # [B, 3, H, W]
        pred_mask = torch.sigmoid(mask_logits)

        return backbone_out, pred_mask


    def infer_topo(self, backbone_out, batch_topo_data):
        device = backbone_out.device if isinstance(backbone_out, torch.Tensor) else backbone_out['backbone_fpn'][0].device
        keypoints, keypoints_valid_mask = (
            batch_topo_data["keypoints"],
            batch_topo_data["valid"],
        )
        keypoints, keypoints_valid_mask = keypoints.to(device), keypoints_valid_mask.to(device)
        
        pred_adj_points_logits = self.topo_decoder(
            backbone_out=backbone_out,
            keypoints=keypoints,
            keypoints_valid_mask=keypoints_valid_mask,
            upsampler=self.feature_upsampler
        )

        pred_adj_points_validity = torch.softmax(pred_adj_points_logits[..., 0:2], dim=-1)[:, :, :, 0]  # -> [B, N_points, num_queries]
        pred_adj_points_coords = torch.tanh(pred_adj_points_logits[..., 2:4])  #  -> [B, N_points, num_queries, 2]

        return pred_adj_points_validity, pred_adj_points_coords


    @torch.no_grad()
    def _visualize(self, rgb, batch, batch_idx, output, keypoints):
        pred_logits, pred_coords = torch.detach(output['pred_logits']), torch.detach(output['pred_coords'])
        gt_coords  = batch['batched_topo_targets']
        # pred_validity: [B, N_points, Num_queries], pred_coords: [B, N_points, Num_queries, 2]
        pred_validity, pred_coords = torch.softmax(pred_logits, dim=-1)[...,0].cpu().numpy(), pred_coords.cpu().numpy()
        
        if batch_idx%100 == 0:
            import cv2 as cv
            max_viz_num = 4
            vis_size = 64
            
            for s_idx, batch_pnt in enumerate(keypoints[:max_viz_num].cpu().numpy()): # 一个batch中的第几个样本
                for idx, pnt in enumerate(batch_pnt):   # 一个batch的一个样本中的第几个kpt
                    vised = False
                    r, c = round(pnt[0]), round(pnt[1])
                    if (r-vis_size>0) and (r+vis_size<self.cfg.PATCH_SIZE) and (c-vis_size>0) and (c+vis_size<self.cfg.PATCH_SIZE):
                        ROI = rgb[s_idx, r-vis_size:r+vis_size, c-vis_size:c+vis_size, :].detach().cpu().numpy()
                        single_point_gt_coords = gt_coords[s_idx][idx]['coords'].cpu().numpy()
                        single_point_pred_validity = pred_validity[s_idx, idx, :]
                        single_point_pred_coords = pred_coords[s_idx, idx, :, :]
                        ROI = cv.cvtColor(ROI, cv.COLOR_RGB2BGR)
                        cv.circle(ROI, (vis_size, vis_size), radius=3, color=(255, 255, 0), thickness=-1) # 橙色中心点
                        
                        for next_pnt_prob, next_pnt_coord in zip(single_point_pred_validity, single_point_pred_coords):
                            if next_pnt_prob > self.cfg.TOPO_DECODER.VALIDITY:
                                next_r, next_c = int(vis_size+self.cfg.NORM_D*next_pnt_coord[0]), int(vis_size+self.cfg.NORM_D*next_pnt_coord[1])
                                cv.circle(ROI, (next_c, next_r), radius=5, color=(203, 192, 255), thickness=-1)   # 粉红色
                                vised = True
                        for gt_next_coord in single_point_gt_coords:
                            gt_next_r, gt_next_c = int(vis_size+self.cfg.NORM_D*gt_next_coord[0]), int(vis_size+self.cfg.NORM_D*gt_next_coord[1])
                            cv.circle(ROI, (gt_next_c, gt_next_r), radius=3, color=(144, 238, 144), thickness=-1)   # 浅绿色
                            
                        if vised:
                            cv.imwrite(f'./vis/{self.cfg.DATASET}_topo/batch_{batch_idx}_sample_{s_idx}_NO{idx}kpt_vis.png', ROI)
                            break   
                        
                        
    @torch.no_grad()
    def _vis_and_update_iou(self, rgb, mask_targets, output, batch_idx):
        pred_mask_logits= output['pred_mask']   # BHW3
        gt_keypoints_mask = torch.stack([single_mask['keypoints_mask'] for single_mask in mask_targets]).unsqueeze(-1)    # BHW -> BHW1
        gt_samplepoints_mask = torch.stack([single_mask['samplepoints_mask'] for single_mask in mask_targets]).unsqueeze(-1) 
        gt_road_mask = torch.stack([single_mask['road_mask'] for single_mask in mask_targets]).unsqueeze(-1) 
        
        gt_mask = torch.cat([gt_keypoints_mask, gt_samplepoints_mask, gt_road_mask], dim=-1) # BHW3
        pred_mask = torch.sigmoid(pred_mask_logits)
        # Log images
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :, :]
            viz_pred_keypoint = pred_mask[:max_viz_num, :, :, 0]
            viz_gt_keypoint = gt_mask[:max_viz_num, :, :, 0]
            
            viz_pred_samplepoint = pred_mask[:max_viz_num, :, :, 1]
            viz_gt_samplepoint = gt_mask[:max_viz_num, :, :, 1]
            
            viz_pred_road = pred_mask[:max_viz_num, :, :, 2]
            viz_gt_road = gt_mask[:max_viz_num, :, :, 2]

            columns = ['rgb', 'gt_keypoint', 'pred_keypoint', 'gt_samplepoint', 'pred_samplepoint', 'gt_road', "pred_road"]
            data = [[wandb.Image(x.cpu().numpy()) for x in row] for row in list(zip(viz_rgb, viz_gt_keypoint, viz_pred_keypoint, viz_gt_samplepoint, viz_pred_samplepoint, viz_gt_road, viz_pred_road))]
            self.logger.log_table(key='viz_table', columns=columns, data=data)
        self.keypoint_iou.update(pred_mask[..., 0], gt_mask[..., 0])
        self.samplepoint_iou.update(pred_mask[..., 1], gt_mask[..., 1])
        self.road_iou.update(pred_mask[..., 2], gt_mask[..., 2])
        
         
    def _set_seed(self, init_value=42, current_epoch=0):   
        seed = init_value + current_epoch
        pl.seed_everything(seed=seed)       # 会打印输出
        
        
    def on_train_epoch_start(self):
        self._set_seed(current_epoch=self.current_epoch)
        # print(f'Training Epoch {self.current_epoch}: Seed set to {42+self.current_epoch}')
    
       
    def training_step(self, batch, batch_idx):
        rgb = batch['rgb']
        keypoints, valid_mask = batch['keypoints'], batch['valid_mask']
        mask_targets, topo_targets = batch['mask_targets'], batch['topo_targets']
        
        output = self(rgb, keypoints, valid_mask)
        loss_dict = self.criterion(output, topo_targets=topo_targets, mask_targets=mask_targets)
        
        keypoint_mask_loss, samplepoint_mask_loss, road_mask_loss = loss_dict['loss_mask']
        prob_loss = loss_dict['loss_prob']
        coord_loss = loss_dict['loss_topo']
        mask_loss = keypoint_mask_loss + samplepoint_mask_loss + road_mask_loss
        
        loss =  self.cfg.CRITERION.MASK_LOSS_COEF*mask_loss + \
                self.cfg.CRITERION.PROB_LOSS_COEF*prob_loss + \
                self.cfg.CRITERION.COORD_LOSS_COEF*coord_loss
            
        if self.cfg.VIS_TRAIN:
            self._visualize(rgb, batch, batch_idx, output, keypoints)
        
        self.log('mask_loss', mask_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log('prob_loss', prob_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log('coord_loss', coord_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log('loss', loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        
        return loss
        
            
    def validation_step(self, batch, batch_idx):
        rgb = batch['rgb']
        keypoints, valid_mask = batch['keypoints'], batch['valid_mask']
        mask_targets, topo_targets = batch['mask_targets'], batch['topo_targets']
        
        output = self(rgb, keypoints, valid_mask)
        loss_dict = self.criterion(output, topo_targets=topo_targets, mask_targets=mask_targets)
        
        keypoint_mask_loss, samplepoint_mask_loss, road_mask_loss = loss_dict['loss_mask']
        prob_loss = loss_dict['loss_prob']
        coord_loss = loss_dict['loss_topo']
        mask_loss = keypoint_mask_loss + samplepoint_mask_loss + road_mask_loss
        
        loss =  self.cfg.CRITERION.MASK_LOSS_COEF*mask_loss + \
                self.cfg.CRITERION.PROB_LOSS_COEF*prob_loss + \
                self.cfg.CRITERION.COORD_LOSS_COEF*coord_loss
        
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_prob_loss', prob_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_coord_loss', coord_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self._vis_and_update_iou(rgb, mask_targets, output, batch_idx)
        self._vis_and_update_iou(rgb, mask_targets, output, batch_idx)
        
        return loss


    def configure_optimizers(self):
        param_dicts = []
        if self.cfg.UNFROZEN_SAM:
            encoder_lr = self.cfg.BASE_LR * self.cfg.ENCODER_LR_FACTOR
            encoder_param = {
                "params": [p for p in self.image_encoder.parameters()],
                "lr": encoder_lr,
            }
            param_dicts.append(encoder_param)
        else:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        mask_decoder_param = {
            "params": [p for p in self.mask_decoder.parameters()],
            "lr": self.cfg.BASE_LR,
        }
        param_dicts.append(mask_decoder_param)
            
        topo_decoder_param = {
            "params": [p for p in self.topo_decoder.parameters()],
            "lr": self.cfg.BASE_LR,
        }
        param_dicts.append(topo_decoder_param)

        # 分别打印参数量
        # for i, param_dict in enumerate(param_dicts):
        #     param_num = sum([int(p.numel()) for p in param_dict["params"]])
        #     print(f"optim param dict {i} params num: {param_num}")

        if self.cfg.DATASET == 'cityscale':
            milestones = [7, 11, 15]
        elif self.cfg.DATASET == 'globalscale':
            milestones = [10, 15]
        elif self.cfg.DATASET == 'spacenet':
            milestones = [10, 20, 25]
        optimizer = torch.optim.AdamW(params=param_dicts, lr=self.cfg.BASE_LR)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, 
            milestones=milestones, 
            gamma=0.1,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        samplepoint_iou = self.samplepoint_iou.compute()
        road_iou = self.road_iou.compute()
        
        self.log("keypoint_iou", keypoint_iou)
        self.log("samplepoint_iou", samplepoint_iou)
        self.log("road_iou", road_iou)
        
        self.keypoint_iou.reset()
        self.samplepoint_iou.reset()
        self.road_iou.reset()

