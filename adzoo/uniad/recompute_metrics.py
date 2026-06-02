#!/usr/bin/env python
"""从已保存的 results.pkl 重新计算评估指标，无需重跑推理。"""
import argparse
from mmcv.fileio.io import load
from mmcv.datasets import build_dataset
from mmcv.utils import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('results_pkl')
    parser.add_argument('--eval', type=str, nargs='+', default=['bbox'])
    parser.add_argument('--lora-stage', type=int, choices=[1, 2], help='Override training stage for evaluation')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))

    outputs = load(args.results_pkl)
    print(f'Loaded {len(outputs)} results from {args.results_pkl}')

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule', 'by_epoch']:
        eval_kwargs.pop(key, None)
    
    lora_stage = args.lora_stage
    if lora_stage is None:
        lora_stage = cfg.model.get('coupled_lora_cfg', {}).get('training_stage', None)
    eval_kwargs['lora_stage'] = lora_stage
    eval_kwargs.update(dict(metric=args.eval))

    print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == '__main__':
    main()
