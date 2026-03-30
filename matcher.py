# matcher

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

# Modified acoording to task

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_coord: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_coord: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, pred_logits, pred_coords, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries: 
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_coords": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_coords] (where num_target_coords is the number of ground-truth
                           objects in the target) containing the class labels
                 "coords": Tensor of dim [num_target_coords, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_coords)
        """
        pred_logits = pred_logits.flatten(0, 1)     # [batch_size*num_points, num_queries, num_classes]
        pred_coords = pred_coords.flatten(0, 1)     # [batch_size*num_points, num_queries, 2]
        bs, num_queries = pred_logits.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = pred_logits.flatten(0, 1).softmax(-1)  # [batch_size*num_points*num_queries, num_classes]
        out_coord = pred_coords.flatten(0, 1)  # [batch_size*num_points*num_queries, 2]
        
        tgt_ids = torch.cat([v["labels"] for v in targets]).to(torch.int) 
        tgt_coord = torch.cat([v["coords"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_coord = torch.cdist(out_coord, tgt_coord, p=2)
        C = self.cost_coord * cost_coord + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["coords"]) for v in targets]
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(set_cost_class, set_cost_coord):
    return HungarianMatcher(cost_class=set_cost_class, cost_coord=set_cost_coord)
