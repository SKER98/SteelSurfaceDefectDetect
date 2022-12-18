import torch
from thop import profile,clever_format

def cal_model_size(model):
	template = torch.randn(1, 3, 224, 224).cuda()
	# list =[{'filename': '../data/NEU-DET/IMAGES/patches_106.jpg', 'ori_filename': 'patches_106.jpg', 'ori_shape': (200, 200, 3), 'img_shape': (224, 224, 3), 'pad_shape': (224, 224, 3), 'scale_factor': array([1.12, 1.12, 1.12, 1.12], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([128.28, 128.28, 128.28], dtype=float32), 'std': array([26.79, 26.79, 26.79], dtype=float32), 'to_rgb': False}, 'batch_input_shape': (224, 224)}]
	flops, params = profile(model, inputs=(template, None))
	print("flops:" + str(flops))
	print("params:" + str(params))
	flops, params = clever_format([flops, params], "%.3f")
	print("flops:" + str(flops))
	print("params:" + str(params))