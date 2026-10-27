#!/usr/bin/env python3
"""
EmberBot — train the YOLOv8 hazmat sign detector.

Pipeline:
  1. setup_dataset()  split the Hazmat13 images/labels into train and validation
                      folders, using the dataset's own manifest files
  2. create_config()  write data.yaml (dataset paths + class names) for Ultralytics
  3. train_yolo()     fine-tune the pretrained YOLOv8-nano model for 20 epochs

Expects the extracted dataset at ./trained-dataset/ with:
  images/, labels/ (YOLO format), hazmat.names,
  manifast_train.txt and manifast_validation.txt (spelling as shipped in the dataset)

Run:  python train.py
"""

import os
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

def setup_dataset():
    """Organize dataset using manifest files."""
    print("Setting up dataset...")
    
    # Create train/val directories
    dirs = ['dataset/images/train', 'dataset/images/val', 'dataset/labels/train', 'dataset/labels/val']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Read manifest files: each line is an image path; keep just the file name
    with open('trained-dataset/manifast_train.txt', 'r') as f:
        train_files = [line.strip().split('/')[-1] for line in f if line.strip()]
    
    with open('trained-dataset/manifast_validation.txt', 'r') as f:
        val_files = [line.strip().split('/')[-1] for line in f if line.strip()]
    
    # Copy each training image and its matching label (same name, .txt) into place
    for filename in train_files:
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # Copy image
            src_img = f'trained-dataset/images/{filename}'
            if os.path.exists(src_img):
                shutil.copy2(src_img, f'dataset/images/train/{filename}')
            
            # Copy label
            label_name = filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            src_label = f'trained-dataset/labels/{label_name}'
            if os.path.exists(src_label):
                shutil.copy2(src_label, f'dataset/labels/train/{label_name}')
    
    # Same again for the validation split
    for filename in val_files:
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            # Copy image
            src_img = f'trained-dataset/images/{filename}'
            if os.path.exists(src_img):
                shutil.copy2(src_img, f'dataset/images/val/{filename}')
            
            # Copy label
            label_name = filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            src_label = f'trained-dataset/labels/{label_name}'
            if os.path.exists(src_label):
                shutil.copy2(src_label, f'dataset/labels/val/{label_name}')
    
    print(f"Copied {len(train_files)} training files and {len(val_files)} validation files")

def create_config():
    """Create data.yaml config file."""
    # Read class names
    with open('trained-dataset/hazmat.names', 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    # Ultralytics reads this file to find the images and the class list
    config = {
        'path': './dataset',
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"Created config with {len(classes)} classes: {classes}")

def train_yolo():
    """Train YOLO model."""
    print("Starting YOLO training...")
    
    # Start from YOLOv8-nano weights pretrained on COCO (transfer learning)
    model = YOLO('yolov8n.pt')  # nano model - fastest
    
    # Fine-tune on Hazmat13
    model.train(
        data='data.yaml',
        epochs=20,
        imgsz=640,      # images resized to 640x640
        batch=150,      # images per training step
        name='hazmat_model',
        device='0' if os.system('nvidia-smi') == 0 else 'cpu'  # use an NVIDIA GPU if present
    )
    
    print("Training complete! Weights saved under runs/detect/hazmat_model*/weights/best.pt — copy best.pt into model/")

def main():
    print("=== Simple YOLO Training ===\n")
    
    # Stop early with a clear message if the dataset hasn't been extracted
    if not os.path.exists('trained-dataset'):
        print("Error: trained-dataset folder not found!")
        print("Make sure you've extracted trained-dataset.zip")
        return
    
    # Setup dataset
    setup_dataset()
    
    # Create config
    create_config()
    
    # Train
    train_yolo()
    
    print("\n=== Done! ===")
    print("Test your model with:")
    print("python test.py your_image.jpg")

if __name__ == "__main__":
    main()