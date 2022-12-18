_base_ = [
	'../_base_/datasets/coco_detection.py',
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
	type='FasterRCNN',
	backbone=dict(
		type='Darknet',
		depth=53,
		out_indices=(2, 3, 4, 5),
		init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
	neck=dict(
		type='FPN',
		in_channels=[128, 256,512,1024],
		out_channels=256,
		num_outs=5),
	rpn_head=dict(
		type='RPNHead',
		in_channels=256,
		feat_channels=256,
		anchor_generator=dict(
			type='AnchorGenerator',
			scales=[8],
			ratios=[0.5, 1.0, 2.0],
			strides=[4, 8, 16, 32, 64]),
		bbox_coder=dict(
			type='DeltaXYWHBBoxCoder',
			target_means=[.0, .0, .0, .0],
			target_stds=[1.0, 1.0, 1.0, 1.0]),
		loss_cls=dict(
			type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
		loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
	roi_head=dict(
		type='StandardRoIHead',
		bbox_roi_extractor=dict(
			type='SingleRoIExtractor',
			roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
			out_channels=256,
			featmap_strides=[4, 8, 16, 32]),
		bbox_head=dict(
			type='Shared2FCBBoxHead',
			in_channels=256,
			fc_out_channels=1024,
			roi_feat_size=7,
			num_classes=6,
			bbox_coder=dict(
				type='DeltaXYWHBBoxCoder',
				target_means=[0., 0., 0., 0.],
				target_stds=[0.1, 0.1, 0.2, 0.2]),
			reg_class_agnostic=False,
			loss_cls=dict(
				type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
			loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
	# model training and testing settings
	train_cfg=dict(
		rpn=dict(
			assigner=dict(
				type='MaxIoUAssigner',
				pos_iou_thr=0.7,
				neg_iou_thr=0.3,
				min_pos_iou=0.3,
				match_low_quality=True,
				ignore_iof_thr=-1),
			sampler=dict(
				type='RandomSampler',
				num=256,
				pos_fraction=0.5,
				neg_pos_ub=-1,
				add_gt_as_proposals=False),
			allowed_border=-1,
			pos_weight=-1,
			debug=False),
		rpn_proposal=dict(
			nms_pre=2000,
			max_per_img=1000,
			nms=dict(type='nms', iou_threshold=0.7),
			min_bbox_size=0),
		rcnn=dict(
			assigner=dict(
				type='MaxIoUAssigner',
				pos_iou_thr=0.5,
				neg_iou_thr=0.5,
				min_pos_iou=0.5,
				match_low_quality=False,
				ignore_iof_thr=-1),
			sampler=dict(
				type='RandomSampler',
				num=512,
				pos_fraction=0.25,
				neg_pos_ub=-1,
				add_gt_as_proposals=True),
			pos_weight=-1,
			debug=False)),
	test_cfg=dict(
		rpn=dict(
			nms_pre=1000,
			max_per_img=1000,
			nms=dict(type='nms', iou_threshold=0.7),
			min_bbox_size=0),
		rcnn=dict(
			score_thr=0.05,
			nms=dict(type='nms', iou_threshold=0.5),
			max_per_img=100)
		# soft-nms is also supported for rcnn testing
		# e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
	))

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
	samples_per_gpu=2,
	workers_per_gpu=0,
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
# load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'