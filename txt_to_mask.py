'''
Utility script that transforms YOLO txt annotations into Binary masks for UNet.
'''

import cv2
import numpy as np
import os
import glob

source_folder = "BTds/valid/labels"
destination_folder = "BTds/valid/masks"
image_size = (640, 640)

os.makedirs(destination_folder, exist_ok=True)
txt_files = glob.glob(os.path.join(source_folder, '*.txt'))

for txt_file in txt_files:
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    with open(txt_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            class_id = int(values[0])  # Ignore class_id in this case
            points = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= image_size[0]
            points[:, 1] *= image_size[1]
            points = points.astype(np.int32)
            cv2.fillPoly(mask, [points], color=255)
            
    base_filename = os.path.basename(txt_file).replace('.txt', '.png')
    mask_path = os.path.join(destination_folder, base_filename)
    cv2.imwrite(mask_path, mask)

print("Masks generated successfully.")
