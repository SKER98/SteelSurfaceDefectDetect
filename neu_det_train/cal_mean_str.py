import numpy as np
import cv2
import os

image_file = "../data/NEU-DET/IMAGES"
R_means = []
G_means = []
B_means = []
R_stds = []
G_stds = []
B_stds = []
all_files = os.listdir(image_file)
for file in all_files:
	if not file.endswith(".jpg"):
		continue
	path = os.path.join(image_file, file)
	im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	opening = cv2.equalizeHist(cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel))
	closing = cv2.equalizeHist(cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel))
	im = np.stack([im, opening, closing],-1)
	im_R = im[:, :, 0]
	im_G = im[:, :, 1]
	im_B = im[:, :, 2]
	im_R_mean = np.mean(im_R)
	im_G_mean = np.mean(im_G)
	im_B_mean = np.mean(im_B)
	im_R_std = np.std(im_R)
	im_G_std = np.std(im_G)
	im_B_std = np.std(im_B)
	R_means.append(im_R_mean)
	G_means.append(im_G_mean)
	B_means.append(im_B_mean)
	R_stds.append(im_R_std)
	G_stds.append(im_G_std)
	B_stds.append(im_B_std)
a = [R_means, G_means, B_means]
b = [R_stds, G_stds, B_stds]
mean = [0, 0, 0]
std = [0, 0, 0]
mean[0] = np.mean(a[0])
mean[1] = np.mean(a[1])
mean[2] = np.mean(a[2])
std[0] = np.mean(b[0])
std[1] = np.mean(b[1])
std[2] = np.mean(b[2])
print('数据集的RGB平均值为\n[{},{},{}]'.format(mean[0], mean[1], mean[2]))
print('数据集的RGB方差为\n[{},{},{}]'.format(std[0], std[1], std[2]))