from xml.dom.minidom import parse
import xml.dom.minidom
import os
import json
# from my_test.image import *

categories = []
categories.append({'supercategory': 'crazing', 'id': 1, 'name': 'crazing'})
categories.append({'supercategory': 'inclusion', 'id': 2, 'name': 'inclusion'})
categories.append({'supercategory': 'patches', 'id': 3, 'name': 'patches'})
categories.append({'supercategory': 'pitted_surface', 'id': 4, 'name': 'pitted_surface'})
categories.append({'supercategory': 'rolled-in_scale', 'id': 5, 'name': 'rolled-in_scale'})
categories.append({'supercategory': 'scratches', 'id': 6, 'name': 'scratches'})

def find_category_id(name: str):
	for cate in categories:
		if name == cate.get('name'):
			return cate.get('id')

num = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

def is_train(name: str, added: bool):
	id = find_category_id(name)
	if num.get(id) >= 90:
		return False
	else:
		if added:
			num[id] += 1
		return True

train_annotations = []
test_annotations = []
train_images = []
test_images = []
path = '../data/NEU-DET/ANNOTATIONS/'
filelist = [path + i for i in os.listdir(path)]
imglist = [i for i in os.listdir('../output/swin')]
image_id = 1
anno_id = 1
for file in filelist:
	if not file.endswith(".xml"):
		continue

	dom = xml.dom.minidom.parse(file)
	root = dom.documentElement

	image_name = root.getElementsByTagName('filename')[0].firstChild.data
	image_width = int(root.getElementsByTagName('size')[0].getElementsByTagName('width')[0].firstChild.data)
	image_height = int(root.getElementsByTagName('size')[0].getElementsByTagName('height')[0].firstChild.data)

	image = {}
	image['id'] = image_id
	image['width'] = image_width
	image['height'] = image_height
	if not image_name.endswith('jpg'):
		image_name = image_name+".jpg"
	image['file_name'] = image_name

	objects = root.getElementsByTagName('object')
	categorie = None
	boxes = []
	for object in objects:
		categorie = object.getElementsByTagName('name')[0].firstChild.data
		pose = object.getElementsByTagName('pose')[0].firstChild.data
		truncated = int(object.getElementsByTagName('truncated')[0].firstChild.data)
		difficult = int(object.getElementsByTagName('difficult')[0].firstChild.data)
		xmin = int(object.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].firstChild.data)
		ymin = int(object.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].firstChild.data)
		xmax = int(object.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].firstChild.data)
		ymax = int(object.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].firstChild.data)
		bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

		area = float(bbox[2]* bbox[3])

		anno = {}
		anno['id'] = anno_id
		anno['image_id'] = image_id
		anno['category'] = categorie
		anno['category_id'] = find_category_id(categorie)
		anno['truncated'] = truncated
		anno['pose'] = pose
		anno['difficult'] = difficult
		anno['bbox'] = bbox
		anno['area'] = area
		anno['iscrowd'] = 0
		anno['segmentation'] = [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]

		boxes.append(anno)
		if is_train(categorie,False):
			train_annotations.append(anno)
		else:
			test_annotations.append(anno)
		anno_id += 1

	if is_train(categorie, True):
		train_images.append(image)
	else:
		test_images.append(image)
	image_id += 1
	# if image_name in imglist:
	# 	draw_image(load_image('../output/swin/' + image_name), boxes)

	# draw_image(load_image('../data/NEU-DET/IMAGES/'+ image_name),boxes)

train_json_dict = {"images": train_images, "annotations": train_annotations, "categories": categories}
test_json_dict = {"images": test_images, "annotations": test_annotations, "categories": categories}

with open('../data/NEU-DET/json_annos/test_split_first.json', 'w') as f:
	string = json.dumps(train_json_dict)
	f.write(string)
with open('../data/NEU-DET/json_annos/train_split_first.json', 'w') as f:
	string = json.dumps(test_json_dict)
	f.write(string)
