# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : preprocess.py
# Copyright (c) Skye-Song. All Rights Reserved

import cv2
from my_test.image import *


# 构建Gabor滤波器
def build_filters():
	filters = []
	ksize = 15  # gabor尺度，6个
	lamda = np.pi # 波长
	for theta in np.arange(0, np.pi, np.pi / 4):  # gabor方向，0°，45°，90°，135°，共四个
		kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
		kern /= 1.5 * kern.sum()
		filters.append(kern)
	return filters


# Gabor特征提取
def getGabor(img, filters):
	res = []  # 滤波结果
	for i in range(len(filters)):
		# res1 = process(img, filters[i])
		accum = np.zeros_like(img)
		for kern in filters[i]:
			fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
			accum = np.maximum(accum, fimg, accum)
		res.append(np.asarray(accum))

	return res[2] + res[3] - res[0] - res[1]  # 返回滤波结果,结果为24幅图，按照gabor角度排列


crazing_file = '../data/NEU-DET/IMAGES/crazing_20.jpg'
inclusion_file = '../data/NEU-DET/IMAGES/inclusion_1.jpg'
patches_file = '../data/NEU-DET/IMAGES/patches_1.jpg'
pitted_surface_file = '../data/NEU-DET/IMAGES/pitted_surface_1.jpg'
rolled_in_scale_file = '../data/NEU-DET/IMAGES/rolled-in_scale_1.jpg'
scratches_file = '../data/NEU-DET/IMAGES/scratches_1.jpg'

img = cv2.imread(pitted_surface_file, cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(np.round(img).astype(np.uint8))
filters = build_filters()
result = getGabor(equ, filters)
# draw_image(result)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# opening = cv2.equalizeHist(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel))
# draw_image(opening)
# closing = cv2.equalizeHist(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel))
# draw_image(closing)

# can = cv2.Canny(img,50,100)
# draw_image(can)

print()
