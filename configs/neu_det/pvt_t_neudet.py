_base_ = [
	'../_base_/models/retinanet_r50_fpn.py',
	'../_base_/datasets/coco_detection.py',
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

load_from = '../checkpoints/pvt_tiny.pth'
model = dict(
	type='RetinaNet',
	backbone=dict(
		_delete_=True,
		type='PyramidVisionTransformer',
		num_layers=[2, 2, 2, 2],
		init_cfg=dict(checkpoint=load_from)),
	bbox_head=dict(
		num_classes=6),
	neck=dict(in_channels=[64, 128, 320, 512]))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)

dataset_type = 'CocoDataset'
data_root = '../data/NEU-DET/'
classes = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches',)

# img_norm_cfg = dict(
# 	mean=[128.28, 129.88, 130.77], std=[26.79, 73.60, 74.66], to_rgb=False)
img_norm_cfg = dict(
	mean=[128.28, 128.28, 128.28], std=[26.79, 26.79, 26.79], to_rgb=False)
train_pipeline = [
	dict(type='LoadImageFromFile', color_type='color'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
	dict(type='LoadImageFromFile', color_type='color'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(224, 224),
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=True),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			dict(type='ImageToTensor', keys=['img']),
			dict(type='Collect', keys=['img']),
		])
]

data = dict(
	samples_per_gpu=2,
	workers_per_gpu=0,
	train=dict(
		img_prefix=data_root + 'IMAGES/',
		classes=classes,
		ann_file=data_root + 'json_annos/train_split_first.json',
		pipeline=train_pipeline),
	val=dict(
		img_prefix=data_root + 'IMAGES/',
		classes=classes,
		ann_file=data_root + 'json_annos/test_split_first.json',
		pipeline=test_pipeline),
	test=dict(
		img_prefix=data_root + 'IMAGES/',
		classes=classes,
		ann_file=data_root + 'json_annos/test_split_first.json',
		pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=100)
