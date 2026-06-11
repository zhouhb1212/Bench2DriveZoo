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
    import gc
    import torch
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _check_gpu_memory(min_free_gb=4):
    if not torch.cuda.is_available():
        return True, float('inf'), float('inf')
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem / (1024 ** 3)
    total_gb = total_mem / (1024 ** 3)
    return free_gb >= min_free_gb, free_gb, total_gb


def _check_system_memory(min_free_gb=8):
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        import re
        m = re.search(r'MemAvailable:\s+(\d+)\s+kB', meminfo)
        if m is None:
            m = re.search(r'MemFree:\s+(\d+)\s+kB', meminfo)
        if m is None:
            return True, float('inf'), float('inf')
        free_kb = int(m.group(1))
        m_total = re.search(r'MemTotal:\s+(\d+)\s+kB', meminfo)
        total_kb = int(m_total.group(1)) if m_total else free_kb
        free_gb = free_kb / (1024 * 1024)
        total_gb = total_kb / (1024 * 1024)
        return free_gb >= min_free_gb, free_gb, total_gb
    except Exception:
        return True, float('inf'), float('inf')


class EarlyStoppingException(Exception):
    """Exception raised when training is terminated early by evaluation hook."""
    pass


def init_early_stopping(hook, early_stopping):
    hook.early_stopping = early_stopping
    if hook.early_stopping is not None:
        hook.patience = hook.early_stopping.get('patience', 3)
        hook.min_delta = hook.early_stopping.get('min_delta', 0.0)
        hook.warmup_iters = hook.early_stopping.get('warmup_iters', 0)
        hook.patience_counter = 0
        hook.best_score = None
        # Ensure save_best is set if not already set
        if getattr(hook, 'save_best', None) is None:
            hook.save_best = hook.early_stopping.get('metric', 'occ/iou_30x30')
            hook.rule = hook.early_stopping.get('rule', 'greater')
        
        # Handle custom metrics mapping manually to bypass MMCV's default _init_rule
        if hook.save_best == 'composite_iou_pq':
            hook.rule = 'greater'
            hook.compare_func = hook.rule_map[hook.rule]
            hook.key_indicator = hook.save_best
        elif hook.save_best == 'plan_l2_avg':
            hook.rule = 'less'
            hook.compare_func = hook.rule_map[hook.rule]
            hook.key_indicator = hook.save_best
        else:
            hook._init_rule(hook.rule, hook.save_best)


def check_early_stopping(hook, runner, key_score):
    if hook.early_stopping is None or key_score is None:
        return False
    current_iter = runner.iter
    if current_iter < hook.warmup_iters:
        runner.logger.info(
            f'[EarlyStopping] Warmup active: iteration {current_iter}/{hook.warmup_iters}. Skipping check.')
        return False

    runner.logger.info(
        f'[EarlyStopping] Checking metric {hook.save_best} (current={key_score:.4f}, best={hook.best_score})')
    if hook.best_score is None:
        hook.best_score = key_score
        hook.patience_counter = 0
        return False

    if hook.rule == 'greater':
        improved = key_score > hook.best_score + hook.min_delta
    else:
        improved = key_score < hook.best_score - hook.min_delta

    if improved:
        runner.logger.info(
            f'[EarlyStopping] Metric improved from {hook.best_score:.4f} to {key_score:.4f}. Resetting patience.')
        hook.best_score = key_score
        hook.patience_counter = 0
        return False
    else:
        hook.patience_counter += 1
        runner.logger.info(
            f'[EarlyStopping] No improvement. Patience counter: {hook.patience_counter}/{hook.patience}.')
        if hook.patience_counter >= hook.patience:
            runner.logger.info(
                f'[EarlyStopping] Early stopping triggered! Best score: {hook.best_score:.4f} at iter {runner.iter + 1}')
            return True
    return False


def evaluate_with_composite(hook, runner, results):
    eval_res = hook.dataloader.dataset.evaluate(
        results, logger=runner.logger, **hook.eval_kwargs)

    for name, val in eval_res.items():
        runner.log_buffer.output[name] = val
    runner.log_buffer.ready = True

    if hook.save_best is not None:
        if hook.key_indicator == 'composite_iou_pq':
            iou = eval_res.get('occ/iou_30x30', 0.0)
            pq = eval_res.get('occ/pq_30x30', 0.0)
            composite_score = 0.5 * iou + 0.5 * pq
            runner.log_buffer.output['composite_iou_pq'] = composite_score
            return composite_score

        if hook.key_indicator == 'plan_l2_avg':
            l2_keys = [f'planning/L2_{i*0.5:.1f}s' for i in range(1, 7)]
            l2_values = [eval_res.get(k, 0.0) for k in l2_keys if k in eval_res]
            if l2_values:
                avg_l2 = sum(l2_values) / len(l2_values)
                runner.log_buffer.output['plan_l2_avg'] = avg_l2
                return avg_l2
            else:
                runner.logger.warning('[EarlyStopping] plan_l2_avg requested but no planning L2 metrics found!')
                return 0.0

        if not eval_res:
            import warnings
            warnings.warn(
                'Since `eval_res` is an empty dict, the behavior to save '
                'the best checkpoint will be skipped in this evaluation.')
            return None

        if hook.key_indicator == 'auto':
            hook._init_rule(hook.rule, list(eval_res.keys())[0])
        return eval_res[hook.key_indicator]

    return None


class EvalHook(BaseEvalHook):

    def __init__(self, *args, early_stopping=None, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)
        init_early_stopping(self, early_stopping)

    def evaluate(self, runner, results):
        return evaluate_with_composite(self, runner, results)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        should_stop = False
        try:
            if not self._should_evaluate(runner):
                return

            _cleanup_memory()
            results = self.test_fn(runner.model, self.dataloader, show=False)
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            _cleanup_memory()

            dataset = getattr(self.dataloader, 'dataset', None)
            scenario_filter = getattr(dataset, 'scenario_filter', None)
            default_min_ram = '2' if scenario_filter is not None else '8'
            min_free_ram_gb = float(
                os.environ.get('B2D_EVAL_MIN_FREE_RAM_GB', default_min_ram))
            ram_ok, free_ram_gb, total_ram_gb = _check_system_memory(min_free_ram_gb)
            if not ram_ok:
                runner.logger.warning(
                    f'[Validation] Low system RAM '
                    f'({free_ram_gb:.1f}GB free / {total_ram_gb:.1f}GB total, '
                    f'need {min_free_ram_gb:.0f}GB), '
                    f'skipping metric computation at epoch {runner.epoch + 1}')
            else:
                key_score = self.evaluate(runner, results)
                if self.save_best:
                    self._save_ckpt(runner, key_score)
                if hasattr(self, 'early_stopping') and self.early_stopping is not None:
                    should_stop = check_early_stopping(self, runner, key_score)
        except EarlyStoppingException:
            raise
        except Exception:
            _cleanup_memory()
            runner.logger.error(
                f'[Validation] Error at epoch {runner.epoch + 1}:\n'
                f'{traceback.format_exc()}')
            runner.logger.warning(
                '[Validation] Validation failed, skipping and continuing '
                'training...')
        if should_stop:
            raise EarlyStoppingException(f"Early stopped at iter {runner.iter + 1}")


class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, early_stopping=None, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)
        init_early_stopping(self, early_stopping)

    def evaluate(self, runner, results):
        return evaluate_with_composite(self, runner, results)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        should_stop = False
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

                dataset = getattr(self.dataloader, 'dataset', None)
                scenario_filter = getattr(dataset, 'scenario_filter', None)
                default_min_ram = '2' if scenario_filter is not None else '8'
                min_free_ram_gb = float(
                    os.environ.get('B2D_EVAL_MIN_FREE_RAM_GB', default_min_ram))
                ram_ok, free_ram_gb, total_ram_gb = _check_system_memory(min_free_ram_gb)
                if not ram_ok:
                    runner.logger.warning(
                        f'[Validation] Low system RAM '
                        f'({free_ram_gb:.1f}GB free / {total_ram_gb:.1f}GB total, '
                        f'need {min_free_ram_gb:.0f}GB), '
                        f'skipping metric computation at epoch {runner.epoch + 1}')
                else:
                    key_score = self.evaluate(runner, results)
                    if self.save_best:
                        self._save_ckpt(runner, key_score)
                    if hasattr(self, 'early_stopping') and self.early_stopping is not None:
                        should_stop = check_early_stopping(self, runner, key_score)
        except EarlyStoppingException:
            raise
        except Exception:
            _cleanup_memory()
            if runner.rank == 0:
                runner.logger.error(
                    f'[Validation] Error at epoch {runner.epoch + 1}:\n'
                    f'{traceback.format_exc()}')
                runner.logger.warning(
                    '[Validation] Validation failed, skipping and continuing '
                    'training...')

        if hasattr(self, 'early_stopping') and self.early_stopping is not None:
            if dist.is_available() and dist.is_initialized():
                stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int32, device='cuda')
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    should_stop = True
            if should_stop:
                raise EarlyStoppingException(f"Early stopped at iter {runner.iter + 1}")


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

    def __init__(self, *args, dynamic_intervals=None, early_stopping=None,  **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)
        init_early_stopping(self, early_stopping)

    def evaluate(self, runner, results):
        return evaluate_with_composite(self, runner, results)


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
        should_stop = False
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

                dataset = getattr(self.dataloader, 'dataset', None)
                scenario_filter = getattr(dataset, 'scenario_filter', None)
                default_min_ram = '2' if scenario_filter is not None else '8'
                min_free_ram_gb = float(
                    os.environ.get('B2D_EVAL_MIN_FREE_RAM_GB', default_min_ram))
                ram_ok, free_ram_gb, total_ram_gb = _check_system_memory(min_free_ram_gb)
                if not ram_ok:
                    runner.logger.warning(
                        f'[Validation] Low system RAM '
                        f'({free_ram_gb:.1f}GB free / {total_ram_gb:.1f}GB total, '
                        f'need {min_free_ram_gb:.0f}GB), '
                        f'skipping metric computation at epoch {runner.epoch + 1}')
                else:
                    key_score = self.evaluate(runner, results)

                    if self.save_best:
                        self._save_ckpt(runner, key_score)
                    if hasattr(self, 'early_stopping') and self.early_stopping is not None:
                        should_stop = check_early_stopping(self, runner, key_score)
        except EarlyStoppingException:
            raise
        except Exception:
            _cleanup_memory()
            if runner.rank == 0:
                runner.logger.error(
                    f'[Validation] Error at epoch {runner.epoch + 1}:\n'
                    f'{traceback.format_exc()}')
                runner.logger.warning(
                    '[Validation] Validation failed, skipping and continuing '
                    'training...')

        if hasattr(self, 'early_stopping') and self.early_stopping is not None:
            if dist.is_available() and dist.is_initialized():
                stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int32, device='cuda')
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    should_stop = True
            if should_stop:
                raise EarlyStoppingException(f"Early stopped at iter {runner.iter + 1}")
