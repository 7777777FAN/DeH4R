
import torch
import torch.nn as nn
import torch.nn.functional as F



class Criterion(nn.Module):
    """ This class computes the loss for DeH4R.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth coords and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and coord)
    """
    def __init__(self, num_classes, matcher, cfg):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = cfg.CRITERION.EOS_COEF
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)


    def loss_labels(self, pred_logits, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_coords]
        """
        src_logits = pred_logits.flatten(0, 1)    # [batch_size*num_points, num_queries, num_classes]
        if indices is not None:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(torch.int64)
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)


    def loss_coords(self, pred_coords, targets, indices):
        """Compute the losses related to the bounding coords, the L2 regression loss targets dicts 
           must contain the key "coords" containing a tensor of dim [nb_target_coords, 2]
           The target coords are expected in format (x, y), normalized by the ROI size.
        """
        pred_coords = pred_coords.flatten(0, 1)     # [batch_size*num_points, num_queries, 2]
        idx = self._get_src_permutation_idx(indices)
        src_coords = pred_coords[idx]   # [num_matched_pnts, 2]
        target_coords = torch.cat([t['coords'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_coords) 
        num_matched_points = max(src_coords.shape[0], 1)    # 避免极端情况除以0（也就是一个batch的所有样本都完全没有有效的点）
        
        # coord_loss = F.l1_loss(src_coords, target_coords, reduction='none').sum() / num_matched_points   # 归一化
        coord_loss = F.mse_loss(src_coords, target_coords, reduction='none').sum() / num_matched_points   # 归一化
        # angle_loss = (1 - F.cosine_similarity(src_coords, target_coords, dim=1)).sum() / num_matched_points
        
        return coord_loss


    def loss_masks(self, outputs, targets):
        """Compute the losses related to the masks
           targets dicts must contain the key "keypoints_mask",
           "samplepoints_mask" and "road_mask" containing a 
           tensor of dim [h, w]
        """
        assert "pred_mask" in outputs
        src_masks = outputs["pred_mask"]    # BHW3
        
        src_keypoint_masks = src_masks[:, :, :, 0]  # BHW
        tgt_keypoint_masks = [t["keypoints_mask"].unsqueeze(0) for t in targets]  # [1, H, W]
        tgt_keypoint_masks = torch.cat(tgt_keypoint_masks, dim=0).to(src_masks)  # BHW
        
        src_samplepoint_masks = src_masks[:, :, :, 1]  # BHW
        tgt_samplepoint_masks = [t["samplepoints_mask"].unsqueeze(0) for t in targets]  # [1, H, W]
        tgt_samplepoint_masks = torch.cat(tgt_samplepoint_masks, dim=0).to(src_masks)  # BHW
        
        src_road_masks = src_masks[:, :, :, 2]  # BHW
        tgt_road_masks = [t["road_mask"].unsqueeze(0) for t in targets]  # [ [1, H, W], ...]
        tgt_road_masks = torch.cat(tgt_road_masks, dim=0).to(src_masks)  # BHW
        
        keypoint_loss = F.binary_cross_entropy_with_logits(src_keypoint_masks, tgt_keypoint_masks)
        samplepoint_loss = F.binary_cross_entropy_with_logits(src_samplepoint_masks, tgt_samplepoint_masks)
        road_loss = F.binary_cross_entropy_with_logits(src_road_masks, tgt_road_masks)
        
        return keypoint_loss, samplepoint_loss, road_loss


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    
    def forward(self, outputs, topo_targets, mask_targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             mask_targets: list of dicts, such that len(mask_targets) == 1, i.e. [{'keypoints_mask':[B,H,W]}].
             topo_targets: list of dicts, such that len(topo_targets) == batch_size*N_points.
              The expected keys in each dict depends on the losses applied, see each loss' doc   
        """
        losses = {'loss_mask':0, 'loss_prob':0, 'loss_coord':0}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs['pred_logits'], outputs['pred_coords'], topo_targets)

        # Compute all the requested losses
        losses['loss_mask'] = self.loss_masks(outputs, mask_targets)
        losses['loss_prob'] = self.loss_labels(outputs['pred_logits'], topo_targets, indices)
        losses['loss_topo'] = self.loss_coords(outputs['pred_coords'], topo_targets, indices)

        return losses
    
