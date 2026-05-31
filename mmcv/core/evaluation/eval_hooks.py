import bisect
import gc
import os
import os.path as osp
import traceback

import torch
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.utils import is_list_of
from torch.nn.modules.batchnorm import _BatchNorm


def _cleanup_memory(runner=None):
    """Release GPU and CPU memory before/after validation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _check_gpu_memory(min_free_gb=4):
    """Check if there's enough free GPU memory for validation.

    Returns (ok: bool, free_gb: float, total_gb: float).
    """
    if not torch.cuda.is_available():
        return True, float('inf'), float('inf')
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem / (1024 ** 3)
    total_gb = total_mem / (1024 ** 3)
    return free_gb >= min_free_gb, free_gb, total_gb


class EvalHook(BaseEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        try:
            if not self._should_evaluate(runner):
                return

            _cleanup_memory()
            results = self.test_fn(runner.model, self.dataloader, show=False)
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self._save_ckpt(runner, key_score)
        except Exception:
            _cleanup_memory()
            runner.logger.error(
                f'[Validation] Error at epoch {runner.epoch + 1}:\n'
                f'{traceback.format_exc()}')
            runner.logger.warning(
                '[Validation] Validation failed, skipping and continuing '
                'training...')


class DistEvalHook(BaseDistEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        try:
            # Synchronization of BatchNorm's buffer (running_mean
            # and running_var) is not supported in the DDP of pytorch,
            # which may cause the inconsistent performance of models in
            # different ranks, so we broadcast BatchNorm's buffers
            # of rank 0 to other ranks to avoid this.
            if self.broadcast_bn_buffer:
                model = runner.model
                for name, module in model.named_modules():
                    if isinstance(module,
                                  _BatchNorm) and module.track_running_stats:
                        dist.broadcast(module.running_var, 0)
                        dist.broadcast(module.running_mean, 0)

            if not self._should_evaluate(runner):
                return

            _cleanup_memory()
            tmpdir = self.tmpdir
            if tmpdir is None:
                tmpdir = osp.join(runner.work_dir, '.eval_hook')

            results = self.test_fn(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect)
            _cleanup_memory()
            if runner.rank == 0:
                print('\n')
                runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
                key_score = self.evaluate(runner, results)

                if self.save_best:
                    self._save_ckpt(runner, key_score)
        except Exception:
            _cleanup_memory()
            if runner.rank == 0:
                runner.logger.error(
                    f'[Validation] Error at epoch {runner.epoch + 1}:\n'
                    f'{traceback.format_exc()}')
                runner.logger.warning(
                    '[Validation] Validation failed, skipping and continuing '
                    'training...')
                
def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None,  **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        try:
            # Synchronization of BatchNorm's buffer (running_mean
            # and running_var) is not supported in the DDP of pytorch,
            # which may cause the inconsistent performance of models in
            # different ranks, so we broadcast BatchNorm's buffers
            # of rank 0 to other ranks to avoid this.
            if self.broadcast_bn_buffer:
                model = runner.model
                for name, module in model.named_modules():
                    if isinstance(module,
                                  _BatchNorm) and module.track_running_stats:
                        dist.broadcast(module.running_var, 0)
                        dist.broadcast(module.running_mean, 0)

            if not self._should_evaluate(runner):
                return

            # ── 验证前内存保护 ──
            _cleanup_memory()

            # 显存预检查：空闲显存不足 4GB 时跳过验证，避免 OOM → SIGKILL
            min_free_gb = float(
                os.environ.get('B2D_EVAL_MIN_FREE_GPU_GB', '4'))
            ok, free_gb, total_gb = _check_gpu_memory(min_free_gb)
            if not ok:
                if runner.rank == 0:
                    runner.logger.warning(
                        f'[Validation] Low GPU memory '
                        f'({free_gb:.1f}GB free / {total_gb:.1f}GB total, '
                        f'need {min_free_gb:.0f}GB), '
                        f'skipping validation at epoch {runner.epoch + 1}')
                return

            tmpdir = self.tmpdir
            if tmpdir is None:
                tmpdir = osp.join(runner.work_dir, '.eval_hook')

            results = self.test_fn(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect)

            # 验证后再次释放内存，清理 test_fn 残留
            _cleanup_memory()

            if runner.rank == 0:
                print('\n')
                runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

                key_score = self.evaluate(runner, results)

                if self.save_best:
                    self._save_ckpt(runner, key_score)
        except Exception:
            # 验证后清理，避免残留 tensor 影响下一 epoch
            _cleanup_memory()
            if runner.rank == 0:
                runner.logger.error(
                    f'[Validation] Error at epoch {runner.epoch + 1}:\n'
                    f'{traceback.format_exc()}')
                runner.logger.warning(
                    '[Validation] Validation failed, skipping and continuing '
                    'training...')
