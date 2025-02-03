'''
Predict masks using YOLO and UNet and visualize the results.
'''

from ultralytics import YOLO
import numpy as np
import os
import torch
import cv2
from unet_model import UNet
import gc
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk


def yolo(image_path, device):
    COLOR_MAP = {'tumor': [255, 255, 255],}
    model_path = 'best.pt'
    model = YOLO(model_path)
    results = model(image_path, show_boxes=False, save=False, imgsz=640, device=device)
    combined_mask = None
    
    for r in results:
        img = np.copy(r.orig_img)
        
        if combined_mask is None:
            combined_mask = np.zeros_like(img)

        for c in r:
            cls = c.boxes.cls.tolist().pop()
            label = c.names[cls]
            color = COLOR_MAP[label]
            object_mask = np.zeros(img.shape[:2], np.uint8)
            if c.masks.xy:
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                if contour.size > 0:
                    cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)
                    combined_mask[object_mask == 255] = color

    return combined_mask

def unet(image_path, device, unet_model):
    COLOR_MAP = [[0, 0, 0], [255, 255, 255]]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = unet_model(image)
    prediction = prediction.cpu().numpy()

    prediction = np.argmax(prediction, axis=1)
    prediction = np.squeeze(prediction)
    result = np.zeros((256, 256, 3), dtype=np.uint8)

    for i, color in enumerate(COLOR_MAP):
        result[prediction == i] = color

    return result

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared!")

def segmentation(files, output_folder, device):
    # Load UNet model once globally and set it to evaluation mode
    weights_path = 'unet.pth'
    unet_model = UNet(input_channels=3, num_classes=2).to(device)
    unet_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    unet_model.eval()

    for path in files:
        unet_result = unet(path, device, unet_model)
        if unet_result.shape[-1] == 3:
            unet_result = cv2.cvtColor(unet_result, cv2.COLOR_RGB2BGR)
        unet_array = cv2.resize(unet_result, (256, 256), interpolation=cv2.INTER_NEAREST)

        yolo_result = yolo(path, device)
        yolo_array = cv2.resize(yolo_result, (256, 256), interpolation=cv2.INTER_NEAREST)

        base_name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_unet.png"), unet_array)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_yolo.png"), yolo_array)

        clear_memory()

def visualize(input_folder, truth_masks, output_folder):
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        print("No images found in the input folder.")
        return
    
    index = 0
    
    def update_image():
        nonlocal index
        base_name = os.path.splitext(files[index])[0]
        input_path = os.path.join(input_folder, files[index])
        mask_path = os.path.join(truth_masks, base_name + ".png")
        yolo_path = os.path.join(output_folder, base_name + "_yolo.png")
        unet_path = os.path.join(output_folder, base_name + "_unet.png")
        
        input_img = Image.open(input_path).resize((256, 256))
        mask_img = Image.open(mask_path).resize((256, 256)) if os.path.exists(mask_path) else Image.new('RGB', (256, 256))
        yolo_img = Image.open(yolo_path).resize((256, 256)) if os.path.exists(yolo_path) else Image.new('RGB', (256, 256))
        unet_img = Image.open(unet_path).resize((256, 256)) if os.path.exists(unet_path) else Image.new('RGB', (256, 256))
        
        input_tk = ImageTk.PhotoImage(input_img)
        mask_tk = ImageTk.PhotoImage(mask_img)
        yolo_tk = ImageTk.PhotoImage(yolo_img)
        unet_tk = ImageTk.PhotoImage(unet_img)
        
        img_labels[0].config(image=input_tk)
        img_labels[0].image = input_tk
        img_labels[1].config(image=mask_tk)
        img_labels[1].image = mask_tk
        img_labels[2].config(image=yolo_tk)
        img_labels[2].image = yolo_tk
        img_labels[3].config(image=unet_tk)
        img_labels[3].image = unet_tk
        
        img_labels[0].config(text=f"Input: {files[index]}")
        
    def next_image():
        nonlocal index
        if index < len(files) - 1:
            index += 1
            update_image()
    
    def prev_image():
        nonlocal index
        if index > 0:
            index -= 1
            update_image()
    
    root = tk.Tk()
    root.title("Segmentation Visualization")
    
    img_labels = []
    labels = ["Input Image", "Ground Truth Mask", "YOLO Prediction", "UNet Prediction"]
    for i, text in enumerate(labels):
        frame = tk.Frame(root, padx=10, pady=10)
        frame.grid(row=0, column=i)
        lbl = Label(frame, text=text, font=("Arial", 14))
        lbl.pack()
        img_label = Label(frame)
        img_label.pack()
        img_labels.append(img_label)
    
    button_frame = tk.Frame(root)
    button_frame.grid(row=1, column=0, columnspan=4, pady=10)
    
    prev_btn = Button(button_frame, text="Previous", command=prev_image, font=("Arial", 12))
    prev_btn.pack(side=tk.LEFT, padx=20)
    
    next_btn = Button(button_frame, text="Next", command=next_image, font=("Arial", 12))
    next_btn.pack(side=tk.RIGHT, padx=20)
    
    update_image()
    root.mainloop()
    
def main():
    # Paths
    output_folder = "view_results/output"
    input_folder = "view_results/images"
    truth_masks = "view_results/masks"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing files from {input_folder} to {output_folder}")
    print(f"Device set to: {device}")

    if os.path.isfile(input_folder):
        files = [input_folder]
    else:
        files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, file))]

    segmentation(files, output_folder, device)
    visualize(input_folder, truth_masks, output_folder)

if __name__ == "__main__":
    main()