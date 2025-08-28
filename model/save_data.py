import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

def save_yolo_result(img_path, detections, class_map, save_dir):
    """
    保存为YOLO格式，每张图片一个txt文件。
    :param img_path: 图片路径
    :param detections: [{'class_name':..., 'bbox':..., 'score':...}, ...]
    :param class_map: {'dog': 0, 'cat': 1, ...}
    :param save_dir: 保存txt的目录
    """
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(save_dir, base + ".txt")
    with open(txt_path, "w") as f:
        for det in detections:
            class_id = class_map.get(det["class_name"], 0)
            bbox = det["bbox"]
            # 归一化
            x_cen = bbox["x_cen"] / w
            y_cen = bbox["y_cen"] / h
            width = bbox["width"] / w
            height = bbox["height"] / h
            f.write(f"{class_id} {x_cen:.6f} {y_cen:.6f} {width:.6f} {height:.6f}\n")

def save_voc_result(img_path, detections, save_dir):
    """
    保存为VOC格式，每张图片一个xml文件。
    :param img_path: 图片路径
    :param detections: [{'class_name':..., 'bbox':..., 'score':...}, ...]
    :param save_dir: 保存xml的目录
    """
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w, c = img.shape
    base = os.path.splitext(os.path.basename(img_path))[0]
    xml_path = os.path.join(save_dir, base + ".xml")

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = os.path.basename(img_path)
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    for det in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = det["class_name"]
        bbox = det["bbox"]
        x_cen, y_cen, width, height = bbox["x_cen"], bbox["y_cen"], bbox["width"], bbox["height"]
        xmin = int(x_cen - width / 2)
        ymin = int(y_cen - height / 2)
        xmax = int(x_cen + width / 2)
        ymax = int(y_cen + height / 2)
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
