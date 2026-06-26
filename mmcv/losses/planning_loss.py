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


@LOSSES.register_module()
class RelativeCollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0, margin=0.0):
        super(RelativeCollisionLoss, self).__init__()
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        self.weight = weight
        self.margin = margin
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum_pred = sdc_traj_all.new_zeros(1, )
        inter_sum_gt = sdc_traj_all.new_zeros(1, )
        
        for i in range(n_futures):
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev 
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                
                # Pred box
                sdc_bev_box_pred = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])
                # GT box
                sdc_bev_box_gt = self.to_corners([sdc_planning_gt[0, i, 0], sdc_planning_gt[0, i, 1], self.w, self.h, sdc_yaw])
                
                corners_b_dev = future_gt_bbox_corners.to(sdc_traj_all.device)
                
                # AABB for obstacles
                xb1 = torch.max(corners_b_dev[:, :, 0], dim=-1)[0]
                yb1 = torch.max(corners_b_dev[:, :, 1], dim=-1)[0]
                xb2 = torch.min(corners_b_dev[:, :, 0], dim=-1)[0]
                yb2 = torch.min(corners_b_dev[:, :, 1], dim=-1)[0]
                
                # Pred collision
                xa1_pred, ya1_pred = torch.max(sdc_bev_box_pred[:, 0]), torch.max(sdc_bev_box_pred[:, 1])
                xa2_pred, ya2_pred = torch.min(sdc_bev_box_pred[:, 0]), torch.min(sdc_bev_box_pred[:, 1])
                xi1_pred, yi1_pred = torch.minimum(xa1_pred, xb1), torch.minimum(ya1_pred, yb1)
                xi2_pred, yi2_pred = torch.maximum(xa2_pred, xb2), torch.maximum(ya2_pred, yb2)
                w_inter_pred = torch.clamp(xi1_pred - xi2_pred, min=0)
                h_inter_pred = torch.clamp(yi1_pred - yi2_pred, min=0)
                inter_sum_pred += torch.sum(w_inter_pred * h_inter_pred)
                
                # GT collision
                xa1_gt, ya1_gt = torch.max(sdc_bev_box_gt[:, 0]), torch.max(sdc_bev_box_gt[:, 1])
                xa2_gt, ya2_gt = torch.min(sdc_bev_box_gt[:, 0]), torch.min(sdc_bev_box_gt[:, 1])
                xi1_gt, yi1_gt = torch.minimum(xa1_gt, xb1), torch.minimum(ya1_gt, yb1)
                xi2_gt, yi2_gt = torch.maximum(xa2_gt, xb2), torch.maximum(ya2_gt, yb2)
                w_inter_gt = torch.clamp(xi1_gt - xi2_gt, min=0)
                h_inter_gt = torch.clamp(yi1_gt - yi2_gt, min=0)
                inter_sum_gt += torch.sum(w_inter_gt * h_inter_gt)
                
        loss = torch.clamp(inter_sum_pred - inter_sum_gt - self.margin, min=0.0)
        return loss * self.weight

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


@LOSSES.register_module()
class PlanningDirectionLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(PlanningDirectionLoss, self).__init__()
        self.weight = weight
        
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        if sdc_traj.dim() == 4:
            sdc_traj = sdc_traj.mean(dim=1)
            
        B, T, _ = sdc_traj.shape
        zero_pos = sdc_traj.new_zeros(B, 1, 2)
        sdc_traj_with_start = torch.cat([zero_pos, sdc_traj[..., :2]], dim=1)
        gt_traj_with_start = torch.cat([zero_pos, gt_sdc_fut_traj[..., :2]], dim=1)
        
        pred_dirs = sdc_traj_with_start[:, 1:] - sdc_traj_with_start[:, :-1]
        gt_dirs = gt_traj_with_start[:, 1:] - gt_traj_with_start[:, :-1]
        
        # 过滤掉地面真实位移过小（如 < 0.1米）的步骤，避免在车辆静止/极慢速时产生无意义/噪声极大的单位向量方向监督
        gt_dist = torch.sqrt(torch.sum(gt_dirs ** 2, dim=-1))
        dir_mask = (gt_dist > 0.1).to(mask.dtype)
        mask = mask * dir_mask
        
        # 使用更稳定的归一化方式代替 F.normalize，防止在预测移动向量极小时梯度爆炸
        pred_dirs_sq = torch.sum(pred_dirs ** 2, dim=-1, keepdim=True)
        pred_dirs_norm_val = torch.sqrt(pred_dirs_sq + 1e-4)
        pred_dirs_norm = pred_dirs / pred_dirs_norm_val
        
        gt_dirs_norm = F.normalize(gt_dirs, p=2, dim=-1, eps=1e-7)
        
        cos_dist = 1.0 - torch.sum(pred_dirs_norm * gt_dirs_norm, dim=-1)
        
        return torch.sum(cos_dist * mask) / (torch.sum(mask) + 1e-5) * self.weight