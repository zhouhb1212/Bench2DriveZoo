#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import contextlib
import torch
import torch.nn as nn
from mmcv.utils import auto_fp16
from mmcv.models import DETECTORS
import copy
import os
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmcv.models.builder import build_head
import re

def _remap_query_to_occ_feat_keys(state_dict, model_has_q2o_lora=True):
    """
    双向兼容 query_to_occ_feat 的 key 格式。

    model_has_q2o_lora=True (模型已注入 LoRA):
        旧格式 → 新格式: layers.N.weight → layers.N.linear.weight
        用于加载未注入 q2o LoRA 的旧 checkpoint。

    model_has_q2o_lora=False (模型未注入 LoRA):
        新格式 → 旧格式: layers.N.linear.weight → layers.N.weight
        丢弃 layers.N.lora_adapter.*
        用于消融实验：用 inject_q2o_feat=False 加载已注入的 ckpt。
    """
    remap = {}
    drops = []

    if model_has_q2o_lora:
        # 旧→新: layers.N.weight → layers.N.linear.weight
        for k in list(state_dict.keys()):
            m = re.match(r'(query_to_occ_feat\.layers\.\d+)\.(weight|bias)$', k)
            if m:
                new_key = f'{m.group(1)}.linear.{m.group(2)}'
                remap[k] = new_key
        for old_key, new_key in remap.items():
            state_dict[new_key] = state_dict.pop(old_key)
    else:
        # 新→旧: layers.N.linear.weight → layers.N.weight, 丢弃 lora_adapter.*
        for k in list(state_dict.keys()):
            m_lin = re.match(r'(query_to_occ_feat\.layers\.\d+)\.linear\.(weight|bias)$', k)
            if m_lin:
                new_key = f'{m_lin.group(1)}.{m_lin.group(2)}'
                remap[k] = new_key
            if 'query_to_occ_feat.layers.' in k and '.lora_adapter.' in k:
                drops.append(k)
        for old_key, new_key in remap.items():
            state_dict[new_key] = state_dict.pop(old_key)
        for k in drops:
            state_dict.pop(k, None)

    return state_dict


@DETECTORS.register_module()
class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """
    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0
        ),
        coupled_lora_cfg=None,
        **kwargs,
    ):
        super(UniAD, self).__init__(**kwargs)
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)

        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}

        # Occupancy-Planning-Motion Coupled LoRA (三阶段训练管理器)
        if coupled_lora_cfg is not None and occ_head is not None and planning_head is not None:
            # 预加载 occ_head / planning_head / motion_head 预训练权重到子模块
            # 必须在 LoRA 注入之前完成，确保 inject_lora_to_linear 的 copy_() 复制的是预训练值
            pretrained_path = coupled_lora_cfg.get('pretrained_path', None)
            if pretrained_path and os.path.exists(pretrained_path):
                ckpt = torch.load(pretrained_path, map_location='cpu')
                state_dict = ckpt.get('state_dict', ckpt)
                occ_state = {k[len('occ_head.'):]: v for k, v in state_dict.items()
                             if k.startswith('occ_head.')}
                plan_state = {k[len('planning_head.'):]: v for k, v in state_dict.items()
                              if k.startswith('planning_head.')}
                motion_state = {k[len('motion_head.'):]: v for k, v in state_dict.items()
                                if k.startswith('motion_head.')}
                if occ_state:
                    self.occ_head.load_state_dict(occ_state, strict=False)
                if plan_state:
                    self.planning_head.load_state_dict(plan_state, strict=False)
                if motion_state and hasattr(self, 'motion_head'):
                    self.motion_head.load_state_dict(motion_state, strict=False)

            from ..dense_heads.planning_head_plugin.mop_coupled_lora import MOPCoupledLoRA
            motion_head_ref = self.motion_head if hasattr(self, 'motion_head') else None
            self.coupled_lora = MOPCoupledLoRA(
                self.occ_head, self.planning_head, coupled_lora_cfg,
                motion_head=motion_head_ref)
            self.coupled_lora.inject()

            # 冻结所有非 LoRA 参数（backbone/BEVFormer/其他 head），
            # 防止被优化器纳入后 weight_decay 逐步衰减预训练权重
            for name, param in self.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
            self.coupled_lora.set_training_stage(
                coupled_lora_cfg.get('training_stage', 1))
            self.coupled_lora.log_trainable_param_count()
        else:
            self.coupled_lora = None

    def train(self, mode=True):
        """重写 train 方法，以保证在整个微调训练期间，所有已冻结参数模块的
        BatchNorm 和 LayerNorm 强制处于 eval() 模式，防止其 running stats 被漂移污染。
        """
        super(UniAD, self).train(mode)
        if mode and self.coupled_lora is not None:
            # 强制将所有 BatchNorm 相关的层置为 eval 模式并再次冻结参数
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
            # 强制将所有 LayerNorm 层置为 eval 模式并再次冻结参数
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None
    
    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    def load_state_dict(self, state_dict, strict=True):
        """支持消融实验中 query_to_occ_feat 键的映射兼容，并在 shape 不匹配时进行过滤加载。"""
        has_q2o_lora = (self.coupled_lora is not None
                        and getattr(self.coupled_lora, 'inject_q2o_feat', False))
        state_dict = _remap_query_to_occ_feat_keys(state_dict,
                                                    model_has_q2o_lora=has_q2o_lora)
        model_state = self.state_dict()
        for key in list(state_dict.keys()):
            if key in model_state and state_dict[key].shape != model_state[key].shape:
                del state_dict[key]
        return super().load_state_dict(state_dict, strict=False)

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, inputs, return_loss=True, rescale=False):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            losses = self.forward_train(**inputs)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(inputs['img_metas']))
            return outputs
        else:
            outputs = self.forward_test(**inputs, rescale=rescale)
            return outputs

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      
                      # Occ_gt
                      gt_segmentation=None,
                      gt_instance=None, 
                      gt_occ_img_is_valid=None,
                      
                      #planning
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      
                      # fut gt for planning
                      gt_future_boxes=None,
                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
            gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
            gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
            gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
            gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
            gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
            gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
            gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
            gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
            gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
            gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
            gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
            gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
            gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
            gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
            sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
            sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
            command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
            gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
            gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
            
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        losses = dict()
        monitoring_losses = dict()  # 冻结 head 的监控 loss，仅用于观察不参与梯度
        len_queue = img.size(1)

        # 各阶段确定哪些 head 需要梯度
        stage1_only = (self.coupled_lora is not None
                       and self.coupled_lora.get_current_stage() == 1)
        stage2_only = (self.coupled_lora is not None
                       and self.coupled_lora.get_current_stage() == 2)

        # 冻结 head（track/map）用 no_grad 执行，节省显存和算力
        # 所有 LoRA stage 均冻结这些基础 head；仅无 LoRA 时正常执行
        frozen_base_ctx = (
            torch.no_grad() if self.coupled_lora is not None
            else contextlib.nullcontext())

        # Motion head 的梯度控制：
        # Stage 1: Motion LoRA 需要梯度（用 motion loss 训练）
        # Stage 2: 冻结 motion head（只训练 OCC LoRA）
        # Stage 3: Motion LoRA 需要梯度（planning loss 回传）
        current_stage = self.coupled_lora.get_current_stage() if self.coupled_lora else 0
        motion_needs_grad = current_stage in (1, 3)
        frozen_motion_ctx = (
            contextlib.nullcontext() if motion_needs_grad or self.coupled_lora is None
            else torch.no_grad())

        with frozen_base_ctx:
            losses_track, outs_track = self.forward_track_train(
                img, gt_bboxes_3d, gt_labels_3d, gt_past_traj,
                gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                l2g_t, l2g_r_mat, img_metas, timestamp)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        monitoring_losses.update(losses_track)

        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)

        bev_embed = outs_track["bev_embed"]
        bev_pos  = outs_track["bev_pos"]

        img_metas = [each[len_queue-1] for each in img_metas]

        outs_seg = dict()
        if self.with_seg_head:
            with frozen_base_ctx:
                losses_seg, outs_seg = self.seg_head.forward_train(
                    bev_embed, img_metas,
                    gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            monitoring_losses.update(losses_seg)

        outs_motion = dict()
        # Forward Motion Head
        if self.with_motion_head:
            with frozen_motion_ctx:
                ret_dict_motion = self.motion_head.forward_train(
                    bev_embed,
                    gt_bboxes_3d, gt_labels_3d,
                    gt_fut_traj, gt_fut_traj_mask,
                    gt_sdc_fut_traj, gt_sdc_fut_traj_mask,
                    outs_track=outs_track, outs_seg=outs_seg)
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos
            losses_motion = ret_dict_motion["losses"]
            losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            # Stage 1 & Stage 3: motion loss 参与反向传播（训练 Motion LoRA）
            # 其他 Stage: motion loss 仅作为监控指标
            if stage1_only or (self.coupled_lora is not None and self.coupled_lora.get_current_stage() == 3):
                losses.update(losses_motion)
            else:
                monitoring_losses.update(losses_motion)

        # Forward Occ Head
        # Stage 1: 跳过（只训练 motion）
        # Stage 2: 执行（OCC LoRA 需要梯度）
        # Stage 3: 跳过（planning 训练不需要 occ loss）
        skip_occ = stage1_only or (current_stage == 3)
        if self.with_occ_head and not skip_occ:
            if outs_motion['track_query'].shape[1] == 0:# avoid 0 track
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]
            losses_occ = self.occ_head.forward_train(
                bev_embed,
                outs_motion,
                gt_inds_list=gt_inds,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)

        # Forward Plan Head
        # Stage 1: 跳过（只训练 motion）
        # Stage 2: 跳过（只训练 OCC）
        # Stage 3: 执行（Planning + Motion LoRA 联合训练）
        skip_planning = stage1_only or stage2_only
        if self.with_planning_head and not skip_planning:
            outs_planning = self.planning_head.forward_train(
                bev_embed, outs_motion, sdc_planning, sdc_planning_mask,
                command, gt_future_boxes)
            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)

        # 精简监控 loss：每个冻结 head 只保留最典型的一个，用于判断特征质量是否稳定
        # 注意：key 中将 'loss' 替换为 'mon'，避免被 _parse_losses 纳入梯度求和；
        #       值用 .detach() 彻底切断计算图，确保不影响反向传播
        kept_prefixes = set()
        for k, v in monitoring_losses.items():
            prefix = k.split('.')[0]  # track / map / motion
            if prefix not in kept_prefixes:
                mon_key = k.replace('loss', 'mon')
                losses[mon_key] = v.detach()
                kept_prefixes.add(prefix)

        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)

        # LoRA 权重变化监控：每 200 iter 输出 lora_B 相对变化量
        if self.training and self.coupled_lora is not None:
            lora_delta = self.coupled_lora.log_lora_delta()
            if lora_delta is not None:
                # key 不含 'loss'，_parse_losses 仅记录为指标不参与梯度求和
                losses['lora_delta'] = torch.tensor(lora_delta, device=img.device)

        return losses
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # planning gt(for evaluation only)
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
 
                     # Occ_gt (for evaluation only)
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     **kwargs,
                    ):
        """Test function
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        
        if self.prev_frame_num > 0:
            if len(self.prev_frame_infos) < self.prev_frame_num:
                self.prev_frame_info = {
                "prev_bev": None,
                "scene_token": None,
                "prev_pos": 0,
                "prev_angle": 0,
            }
            else:
                self.prev_frame_info = self.prev_frame_infos.pop(0)

        is_first_frame = False
        if self.prev_frame_info['scene_token'] is None or img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            is_first_frame = True
            self.prev_frame_info['prev_bev'] = None
            if self.prev_frame_num > 0:
                self.prev_frame_infos = []
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # first frame
        if is_first_frame:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # following frames
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        


        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]
        
        if self.prev_frame_num > 0:        
            self.prev_frame_infos.append(self.prev_frame_info)        
        
        

        if self.with_seg_head:
            result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed, 
                outs_motion,
                no_query = occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ
        
        if self.with_planning_head:
            planning_gt=dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            if 'pts_bbox' in result_seg[0]:
                for k, v in result_seg[0]['pts_bbox'].items():
                    result_seg[0][k] = v
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head and os.environ.get('ENABLE_PLOT_MODE', None) is None:
            result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
                pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])
        
        for i, res in enumerate(result):
            #res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result


def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
