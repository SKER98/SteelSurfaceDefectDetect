from mmdet.apis import init_detector, inference_detector, show_result_pyplot

# faster_rcnn
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# yolo v3
config_file = 'configs/yolo/yolov3_d53_320_273e_coco.py'
checkpoint_file = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'

device = 'cuda:1'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
# image_path = 'demo/neu-det/crazing_1.jpg'
image_path = 'demo/neu-det/inclusion_1.jpg'
# image_path = 'demo/neu-det/patches_1.jpg'
# image_path = 'demo/neu-det/pitted_surface_1.jpg'
# image_path = 'demo/neu-det/rolled-in_scale_1.jpg'
# image_path = 'demo/neu-det/scratches_1.jpg'
result = inference_detector(model, image_path)
show_result_pyplot(model, image_path,result)

print()
