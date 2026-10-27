"""
Convert Pascal VOC (.xml) annotations from the Hazmat13 dataset into YOLO label files.

YOLO expects one .txt file per image, with one line per object:
    <class_id> <x_center> <y_center> <width> <height>
where all four box values are normalised to 0–1 by the image size.

Usage: place this script next to the extracted `hazmat13/` folder and run
    python convert_voc_to_yolo.py
"""

import os
import xml.etree.ElementTree as ET

XML_DIR = "hazmat13/annotations"      # input: VOC .xml files
LABEL_DIR = "hazmat13/labels"         # output: YOLO .txt files
NAMES_PATH = "hazmat13/hazmat.names"  # one class name per line

os.makedirs(LABEL_DIR, exist_ok=True)

# Load class names; a class's line number becomes its YOLO class id
with open(NAMES_PATH) as f:
    class_names = [c.strip() for c in f if c.strip()]


def convert_bbox(w, h, xmin, ymin, xmax, ymax):
    """Turn corner coordinates in pixels into a normalised centre/size box."""
    xcenter = ((xmin + xmax) / 2) / w
    ycenter = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return xcenter, ycenter, bw, bh


for xml_file in os.listdir(XML_DIR):
    if not xml_file.endswith(".xml"):
        continue

    root = ET.parse(os.path.join(XML_DIR, xml_file)).getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in class_names:
            continue  # skip labels that aren't in our class list
        cls_id = class_names.index(cls_name)
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)

        x, y, bw, bh = convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax)
        yolo_lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

    # Write the YOLO label file with the same base name as the annotation
    out_path = os.path.join(LABEL_DIR, os.path.splitext(xml_file)[0] + ".txt")
    with open(out_path, "w") as f:
        f.write("\n".join(yolo_lines))
