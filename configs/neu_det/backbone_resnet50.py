_base_ = [
	'../_base_/models/faster_rcnn_r50_fpn.py',
	'../_base_/datasets/coco_detection.py',
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
	roi_head=dict(
		bbox_head=dict(num_classes=6)))

dataset_type = 'CocoDataset'
data_root = '../data/NEU-DET/'
classes = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches',)

img_norm_cfg = dict(
	mean=[128.28, 128.28, 128.28], std=[26.79, 26.79, 26.79], to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
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
	train=dict(
		img_prefix=data_root +'IMAGES/',
		classes=classes,
		ann_file=data_root +'json_annos/train_split_first.json',
		pipeline=train_pipeline),
	val=dict(
		img_prefix=data_root +'IMAGES/',
		classes=classes,
		ann_file=data_root +'json_annos/test_split_first.json',
		pipeline=test_pipeline),
	test=dict(
		img_prefix=data_root +'IMAGES/',
		classes=classes,
		ann_file=data_root +'json_annos/test_split_first.json',
		pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=100)
load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'