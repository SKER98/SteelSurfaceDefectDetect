_base_ = '../yolof/yolof_r50_c5_8x8_1x_coco.py'

model = dict(
	bbox_head=dict(num_classes=6))

dataset_type = 'CocoDataset'
data_root = '../data/NEU-DET/'
classes = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches',)

data = dict(
	samples_per_gpu=4,
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
load_from = '../checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'