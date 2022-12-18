# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import pickle
import warnings

from mmcv import Config, DictAction
from mmdet.datasets import (build_dataset,replace_ImageToTensor)

def parse_args():
	config_file = '../configs/neu_det/swint_3x_neudet.py'
	checkpoint_file = '../output/final_epoch/swin.pth'
	out_file = '../output/final_result/ssd.pkl'

	# config_file = '../configs/yolo/yolov3_d53_320_273e_coco.py'
	# checkpoint_file = '../checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'
	# out_file = '../output/yolov3_coco.pkl'

	parser = argparse.ArgumentParser(
		description='MMDet test (and eval) a model')
	parser.add_argument('--config', default=config_file, help='test config file path')
	parser.add_argument('--checkpoint', default=checkpoint_file, help='checkpoint file')
	parser.add_argument(
		'--work-dir',
		help='the directory to save the file containing evaluation metrics')
	parser.add_argument('--out', default=out_file, help='output result file in pickle format')
	parser.add_argument(
		'--eval',
		type=str,
		# default='proposal_fast',
		default='bbox',
		nargs='+',
		help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
		     ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
	parser.add_argument(
		'--options',
		nargs='+',
		action=DictAction,
		help='custom options for evaluation, the key-value pair in xxx=yyy '
		     'format will be kwargs for dataset.evaluate() function (deprecate), '
		     'change to --eval-options instead.')
	parser.add_argument(
		'--eval-options',
		nargs='+',
		action=DictAction,
		help='custom options for evaluation, the key-value pair in xxx=yyy '
		     'format will be kwargs for dataset.evaluate() function')
	args = parser.parse_args()

	if args.options and args.eval_options:
		raise ValueError(
			'--options and --eval-options cannot be both '
			'specified, --options is deprecated in favor of --eval-options')
	if args.options:
		warnings.warn('--options is deprecated in favor of --eval-options')
		args.eval_options = args.options

	return args


def main():
	args = parse_args()
	cfg = Config.fromfile(args.config)
	cfg.data.test.test_mode = True
	samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
	if samples_per_gpu > 1:
		# Replace 'ImageToTensor' to 'DefaultFormatBundle'
		cfg.data.test.pipeline = replace_ImageToTensor(
			cfg.data.test.pipeline)
	# build the dataloader
	dataset = build_dataset(cfg.data.test)
	pkl_file = open(args.out, 'rb')
	outputs = pickle.load(pkl_file)

	kwargs = {} if args.eval_options is None else args.eval_options
	if args.eval:
		eval_kwargs = cfg.get('evaluation', {}).copy()
		# hard-code way to remove EvalHook args
		for key in [
			'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
			'rule'
		]:
			eval_kwargs.pop(key, None)
		eval_kwargs.update(dict(metric=args.eval, **kwargs))
		metric = dataset.evaluate(outputs, **eval_kwargs)
		print(metric)

if __name__ == '__main__':
	main()
