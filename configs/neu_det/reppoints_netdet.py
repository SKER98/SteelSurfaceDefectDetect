_base_ = [
	'../_base_/datasets/coco_detection.py',
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg_1 = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
	type='RepPointsDetector',
	backbone=dict(
		type='ResNet',
		depth=50,
		num_stages=4,
		out_indices=(0, 1, 2, 3),
		frozen_stages=1,
		norm_cfg=dict(type='BN', requires_grad=True),
		norm_eval=True,
		style='pytorch',
		init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
	neck=dict(
		type='FPN',
		norm_cfg=norm_cfg_1,
		in_channels=[256, 512, 1024, 2048],
		out_channels=256,
		start_level=1,
		add_extra_convs='on_input',
		num_outs=5),
	bbox_head=dict(
		type='RepPointsHead',
		num_classes=6,
		in_channels=256,
		feat_channels=256,
		point_feat_channels=256,
		stacked_convs=3,
		num_points=9,
		gradient_mul=0.1,
		point_strides=[8, 16, 32, 64, 128],
		point_base_scale=4,
		norm_cfg=norm_cfg_1,
		loss_cls=dict(
			type='FocalLoss',
			use_sigmoid=True,
			gamma=2.0,
			alpha=0.25,
			loss_weight=1.0),
		loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
		loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
		transform_method='moment'),
	# training and testing settings
	train_cfg=dict(
		init=dict(
			assigner=dict(type='PointAssigner', scale=4, pos_num=1),
			allowed_border=-1,
			pos_weight=-1,
			debug=False),
		refine=dict(
			assigner=dict(
				type='MaxIoUAssigner',
				pos_iou_thr=0.5,
				neg_iou_thr=0.4,
				min_pos_iou=0,
				ignore_iof_thr=-1),
			allowed_border=-1,
			pos_weight=-1,
			debug=False)),
	test_cfg=dict(
		nms_pre=1000,
		min_bbox_size=0,
		score_thr=0.05,
		nms=dict(type='nms', iou_threshold=0.5),
		max_per_img=100))
optimizer = dict(lr=0.01)


dataset_type = 'CocoDataset'
data_root = '../data/NEU-DET/'
classes = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches',)

data = dict(
	samples_per_gpu=2,
	workers_per_gpu=0,
	train=dict(
		img_prefix=data_root + 'IMAGES/',
		classes=classes,
		ann_file=data_root + 'json_annos/train_split_first.json'),
	val=dict(
		img_prefix=data_root + 'IMAGES/',
		classes=classes,
		ann_file=data_root + 'json_annos/test_split_first.json'),
	test=dict(
		img_prefix=data_root + 'IMAGES/',
		classes=classes,
		ann_file=data_root + 'json_annos/test_split_first.json'))

runner = dict(type='EpochBasedRunner', max_epochs=100)
load_from = '../checkpoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco_20200329-4b38409a.pth'