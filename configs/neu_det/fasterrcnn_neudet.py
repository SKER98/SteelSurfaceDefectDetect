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

data = dict(
	train=dict(
		img_prefix=data_root +'IMAGES/',
		classes=classes,
		ann_file=data_root +'json_annos/train_split_first.json'),
	val=dict(
		img_prefix=data_root +'IMAGES/',
		classes=classes,
		ann_file=data_root +'json_annos/test_split_first.json'),
	test=dict(
		img_prefix=data_root +'IMAGES/',
		classes=classes,
		ann_file=data_root +'json_annos/test_split_first.json'))

runner = dict(type='EpochBasedRunner', max_epochs=100)
load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'