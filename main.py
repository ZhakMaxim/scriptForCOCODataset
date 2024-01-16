from pycocotools.coco import COCO
import requests
import os
import json
import cv2
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

annotation_file_path = 'D:\PyCharm\PycharmProjects\ScriptForDataset\Annotations\instances_train2017.json'

coco = COCO(annotation_file_path)

transport_class_ids = [2, 3, 5, 6, 7, 8]

output_folder = 'transport_data'
os.makedirs(output_folder, exist_ok=True)

# Create a Folder for COCO Annotations
coco_annotations_output_folder = os.path.join(output_folder, 'coco_annotations')
os.makedirs(coco_annotations_output_folder, exist_ok=True)
coco_annotations_output_path = os.path.join(coco_annotations_output_folder, 'coco_annotations.json')

# Create a Folder for YOLO Annotations
yolo_annotations_output_folder = os.path.join(output_folder, 'yolo_annotations')
os.makedirs(yolo_annotations_output_folder, exist_ok=True)

# Create a folder for visualizing annotations
#visualization_output_folder = os.path.join(output_folder, 'visualization')
#os.makedirs(visualization_output_folder, exist_ok=True)

transport_image_ids = []
yolo_annotations = []
coco_annotations = []

for class_id in transport_class_ids:
    img_ids = coco.getImgIds(catIds=class_id)
    transport_image_ids.extend(img_ids)

print(f"Number of photos found : {len(transport_image_ids)}")

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

#Download images with transport and save annotations
for img_info in coco.loadImgs(transport_image_ids):
    img_url = img_info['coco_url']
    img_name = os.path.basename(img_url)
    img_path = os.path.join(output_folder, img_name)


    response = session.get(img_url)
    with open(img_path, 'wb') as f:
        f.write(response.content)

    #Visualization of annotations in the image
    #image = cv2.imread(img_path)

    img_id = img_info['id']
    img_annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

    #Creating annotations in YOLO and COCO format
    for annotation in img_annotations:
        if annotation['category_id'] in transport_class_ids:
            class_id = annotation["category_id"]
            #color = (0, 0, 255)
            bbox = annotation["bbox"]
            x, y, w, h = [int(val) for val in bbox]
            #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            image_width = img_info['width']
            image_height = img_info['height']

            #Annotations in YOLO format
            yolo_annotation = {
                "class_id": class_id,
                "center_x": (bbox[0] + bbox[2] / 2) / image_width,
                "center_y": (bbox[1] + bbox[3] / 2) / image_height,
                "width": bbox[2] / image_width,
                "height": bbox[3] / image_height
            }
            yolo_annotations.append(yolo_annotation)

            #Annotations in COCO format
            coco_annotation = {
                "id": len(coco_annotations) + 1,  # Уникальный ID
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            }
            coco_annotations.append(coco_annotation)

    #Save annotations in YOLO format
    yolo_annotation_path = os.path.join(yolo_annotations_output_folder, img_name.replace(".jpg", ".txt"))
    with open(yolo_annotation_path, 'w') as yolo_output_file:
        for yolo_annotation in yolo_annotations:
            yolo_output_file.write(f"{yolo_annotation['class_id']} "
                                   f"{yolo_annotation['center_x']} {yolo_annotation['center_y']} "
                                   f"{yolo_annotation['width']} {yolo_annotation['height']}\n")

    #Saving an image with visualized marking
    #visualization_image_path = os.path.join(visualization_output_folder, img_name)
    #cv2.imwrite(visualization_image_path, image)


#Save annotations in COCO format
with open(coco_annotations_output_path, 'w') as coco_output_file:
    json.dump(coco_annotations, coco_output_file)

print(f"Извлечено {len(transport_image_ids)} изображений и сохранено в форматах COCO и YOLO.")
