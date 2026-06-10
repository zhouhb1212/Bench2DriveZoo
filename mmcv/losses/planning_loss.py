#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmcv.models import LOSSES


@LOSSES.register_module()
class PlanningLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err + 1e-7, exponent=0.5)
        return torch.sum(err * mask)/(torch.sum(mask) + 1e-5)


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        super(CollisionLoss, self).__init__()
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        self.weight = weight
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(1, )
        dump_sdc = []
        for i in range(n_futures):
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev 
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])
                dump_sdc.append(sdc_bev_box.cpu().detach().numpy())
                
                # 向量化并行计算碰撞面积，避免 CPU-GPU 同步瓶颈
                corners_b_dev = future_gt_bbox_corners.to(sdc_traj_all.device)
                xa1, ya1 = torch.max(sdc_bev_box[:, 0]), torch.max(sdc_bev_box[:, 1])
                xa2, ya2 = torch.min(sdc_bev_box[:, 0]), torch.min(sdc_bev_box[:, 1])
                
                xb1 = torch.max(corners_b_dev[:, :, 0], dim=-1)[0]
                yb1 = torch.max(corners_b_dev[:, :, 1], dim=-1)[0]
                xb2 = torch.min(corners_b_dev[:, :, 0], dim=-1)[0]
                yb2 = torch.min(corners_b_dev[:, :, 1], dim=-1)[0]
                
                xi1, yi1 = torch.minimum(xa1, xb1), torch.minimum(ya1, yb1)
                xi2, yi2 = torch.maximum(xa2, xb2), torch.maximum(ya2, yb2)
                
                w_inter = torch.clamp(xi1 - xi2, min=0)
                h_inter = torch.clamp(yi1 - yi2, min=0)
                inter_sum += torch.sum(w_inter * h_inter)
        return inter_sum * self.weight

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        corners = torch.tensor([
            [w/2, -l/2], [w/2, l/2], [-w/2, l/2], [-w/2,-l/2]  
        ]).to(x.device) # 4,2
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        rot_mat = torch.stack([
            torch.stack([cos_t, sin_t]),
            torch.stack([-sin_t, cos_t])
        ]).to(x.device)
        
        translation = torch.stack([x, y])[:, None]
        new_corners = rot_mat @ corners.T + translation
        return new_corners.T