# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : image.py
# Copyright (c) Skye-Song. All Rights Reserved

import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import math
import json


def load_image(img_file: str) -> np.array:
	img = cv2.imread(img_file, cv2.IMREAD_COLOR)
	return img


def draw_image(img, boxes=None, color='g'):
	img = np.array(img, dtype=int)
	fig, ax = plt.subplots(1)
	ax.imshow(img)
	if boxes is not None:
		if isinstance(boxes, np.ndarray):
			boxes = list(np.array(boxes, dtype=int))

		# for box in boxes:
		# 	rect = patches.Rectangle(box[:2], box[2]-box[0], box[3] - box[1], linewidth=2, fill=False, edgecolor=color)
		# 	ax.add_patch(rect)
		for anno in boxes:
			box = anno['bbox']
			categorie = anno['category']
			truncated = anno['truncated']
			pose = anno['pose']
			difficult = anno['difficult']
			rect = patches.Rectangle(box[:2], box[2], box[3], linewidth=4, fill=False, edgecolor=color)
			# ax.annotate(categorie, (box[0]+2, box[1]+8), color='r', weight='bold', fontsize=10)
			# ax.annotate('trun:'+str(truncated), (box[0] + 2, box[1] + 16), color='r', weight='bold', fontsize=10)
			# ax.annotate('pose:' + str(pose), (box[0] + 2, box[1] + 24), color='r', weight='bold', fontsize=10)
			# ax.annotate('diff:' + str(difficult), (box[0] + 2, box[1] + 32), color='r', weight='bold', fontsize=10)
			ax.add_patch(rect)
	plt.show()


def draw_norm_image(img, box=None, color='r'):
	img = np.array(img, dtype=int)

	std = [0.229, 0.224, 0.225]
	mean = [0.485, 0.456, 0.406]
	img = img * std + mean
	img = img * 255 / 0.2

	fig, ax = plt.subplots(1)
	ax.imshow(img)
	if box is not None:
		rect = patches.Rectangle(box[:2], box[2], box[3], linewidth=2, fill=False, edgecolor=color)
		ax.add_patch(rect)
	plt.show()


def draw_tensor(a: torch.Tensor):
	a_np = a.squeeze().cpu().clone().detach().numpy()
	if a_np.ndim == 3:
		a_np = np.transpose(a_np, (1, 2, 0))

	# img = a_np * 127 + 127
	fig, ax = plt.subplots(1)
	ax.imshow(a_np)
	plt.show()


def draw_feat(a: torch.Tensor, fix_max_min=True):
	# a = (H,W,channel)
	one_channel = False
	if len(a.squeeze().size()) == 2:
		H, W = a.squeeze().size()
		C = 1
		one_channel = True
	elif len(a.squeeze().size()) == 3:
		H, W, C = a.squeeze().size()
	else:
		return

	a_np = a.squeeze().cpu().clone().detach().numpy()

	if fix_max_min:
		max = 1.5
		min = -1.5
	else:
		max = np.max(a_np)
		min = np.min(a_np)

	a_np = (np.minimum(np.maximum(a_np, min), max) - min) / (max - min) * 255.
	a_np = a_np.astype(int)

	h_num = int(np.ceil(math.sqrt(C)))

	image = np.zeros([H * h_num + h_num - 1, W * h_num + h_num - 1])

	flag = True
	for i in range(h_num):
		for j in range(h_num):
			index = i * h_num + j
			if index >= C:
				flag = False
				break
			if one_channel:
				image[i * (H + 1):i * (H + 1) + H, j * (W + 1):j * (W + 1) + W] = a_np
			else:
				image[i * (H + 1):i * (H + 1) + H, j * (W + 1):j * (W + 1) + W] = a_np[:, :, index]
		if not flag:
			break

	draw_image(image)
