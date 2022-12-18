import json
coco_json = "../data/coco/annotations/instances_val2017.json"
# coco_json = "../data/NEU-DET/json_annos/train_split.json"

json_file = open(coco_json, 'rb')
data = json.load(json_file)
print( )