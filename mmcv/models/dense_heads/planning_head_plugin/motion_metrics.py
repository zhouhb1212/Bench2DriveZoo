"""Agent Motion evaluation metrics for trajectory prediction.

Computes min_ADE, min_FDE and Miss Rate for tracked agents (excluding SDC),
following the NuScenes MotionEval protocol.

Usage in validation loop:
    metric = AgentMotionMetric(n_future=12, miss_threshold=2.0).cuda()
    for batch in val_loader:
        metric.update(pred_trajs, gt_trajs, gt_masks)
    result = metric.compute()
"""
import torch
import torch.nn as nn
from torchmetrics import Metric


class AgentMotionMetric(Metric):
    """Agent motion trajectory evaluation metric (excluding SDC).

    Computes:
    - min_ADE: Minimum Average Displacement Error across modes
    - min_FDE: Minimum Final Displacement Error across modes
    - Miss Rate: Fraction of agents where best mode's max displacement > threshold

    Expected inputs:
    - pred_trajs: (N, num_modes, T, 2) - multi-modal predicted trajectories per agent
    - gt_trajs: (N, T, 2) - ground truth future trajectory per agent
    - gt_masks: (N, T) - validity mask (1=valid, 0=invalid timestep)

    where N = total number of matched agents in this batch (variable per frame).
    """

    def __init__(self, n_future=12, miss_threshold=2.0):
        super().__init__()
        self.n_future = n_future
        self.miss_threshold = miss_threshold

        # Accumulated metrics
        self.add_state("ade_sum", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fde_sum", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("miss_count", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, pred_trajs, gt_trajs, gt_masks):
        """
        Args:
            pred_trajs: (N, num_modes, T, 2+) predicted trajectories for N agents
            gt_trajs: (N, T, 2) ground truth trajectories
            gt_masks: (N, T) mask (1=valid timestep, 0=invalid)
        """
        if pred_trajs.numel() == 0 or gt_trajs.numel() == 0:
            return

        # Debug: print shapes on first call
        if self.total.item() == 0:
            print(f"[MotionMetric] pred_trajs.shape={pred_trajs.shape}, "
                  f"gt_trajs.shape={gt_trajs.shape}, gt_masks.shape={gt_masks.shape}")

        # Handle various pred_trajs shapes:
        # Expected: (N, num_modes, T, feat) where feat >= 2
        # But could be (N, T, feat) if only 1 mode, or other shapes
        if pred_trajs.dim() == 3:
            # (N, T, feat) -> add mode dim: (N, 1, T, feat)
            pred_trajs = pred_trajs.unsqueeze(1)

        N, num_modes, T_pred, feat = pred_trajs.shape
        T_gt = gt_trajs.shape[1] if gt_trajs.dim() >= 2 else 0
        T = min(T_pred, self.n_future, T_gt)

        # Only take xy coordinates
        pred_trajs = pred_trajs[:, :, :T, :2]  # (N, modes, T, 2)
        gt_trajs = gt_trajs[:, :T, :2]  # (N, T, 2)
        gt_masks = gt_masks[:, :T]  # (N, T)

        # Expand GT for broadcasting: (N, 1, T, 2)
        gt_expanded = gt_trajs.unsqueeze(1)
        # Per-timestep L2 distance: (N, num_modes, T)
        dist = torch.sqrt(((pred_trajs - gt_expanded) ** 2).sum(dim=-1) + 1e-8)

        # Mask invalid timesteps
        valid = gt_masks.unsqueeze(1)  # (N, 1, T)
        valid_count = valid.sum(dim=-1, keepdim=True).clamp(min=1)  # (N, 1, 1)

        # --- min_ADE: average over valid timesteps, then pick best mode ---
        ade_per_mode = (dist * valid).sum(dim=-1) / valid_count.squeeze(-1)  # (N, num_modes)
        min_ade, best_mode_idx = ade_per_mode.min(dim=1)  # (N,)

        # --- min_FDE: displacement at last valid timestep, best mode ---
        # Find last valid timestep index for each agent
        lengths = gt_masks.sum(dim=1).long().clamp(min=1)  # (N,)
        last_idx = (lengths - 1).clamp(max=T - 1)  # (N,)
        # Gather FDE from best mode
        best_dist = dist[torch.arange(N, device=dist.device), best_mode_idx]  # (N, T)
        min_fde = best_dist[torch.arange(N, device=dist.device), last_idx]  # (N,)

        # --- Miss Rate: max displacement of best mode > threshold ---
        # Set invalid timesteps to -inf so they don't affect max
        masked_dist = best_dist.clone()
        masked_dist[~gt_masks.bool()] = -1.0
        max_dist_per_agent = masked_dist.max(dim=1).values  # (N,)
        miss = (max_dist_per_agent > self.miss_threshold).float()

        # Accumulate
        self.ade_sum += min_ade.sum()
        self.fde_sum += min_fde.sum()
        self.miss_count += miss.sum()
        self.total += N

    def compute(self):
        """Returns dict of motion metrics."""
        total = self.total.clamp(min=1)
        return {
            'min_ade': (self.ade_sum / total).item(),
            'min_fde': (self.fde_sum / total).item(),
            'miss_rate': (self.miss_count / total).item(),
            'num_agents': int(self.total.item()),
        }
