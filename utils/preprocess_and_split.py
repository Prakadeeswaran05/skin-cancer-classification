import os
import shutil
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from image_processing import preprocessing
import argparse

def process_and_split_data(input_root, output_root, val_split=0.1):
    """
    Process and split skin lesion images into training and validation sets.
    
    Args:
        input_root: Root directory of the original dataset
        output_root: Root directory to save processed images
        val_split: Proportion of images to use for validation
    """
    categories = ['Benign', 'Malignant']
    
    for category in categories:
        input_folder = os.path.join(input_root, 'train', category)
        output_train_folder = os.path.join(output_root, 'train', category)
        output_val_folder = os.path.join(output_root, 'val', category)     
        os.makedirs(output_train_folder, exist_ok=True)
        os.makedirs(output_val_folder, exist_ok=True)   
        image_paths = glob(os.path.join(input_folder, '*.jpg'))  
        
        print(f"Found {len(image_paths)} images in {input_folder}")   
        # Split into train and validation sets
        train_paths, val_paths = train_test_split(image_paths, test_size=val_split, random_state=42)
        
        
        for img_path in train_paths:
            img_processed = preprocessing(img_path)
            output_path = os.path.join(output_train_folder, os.path.basename(img_path))
            cv2.imwrite(output_path, img_processed)
            print(f"Saved processed image to {output_path}")
        
        for img_path in val_paths:
            img_processed = preprocessing(img_path)
            output_path = os.path.join(output_val_folder, os.path.basename(img_path))
            cv2.imwrite(output_path, img_processed)
            print(f"Saved processed image to {output_path}")
        
        print(f"Total processed images in {output_train_folder}: {len(glob(os.path.join(output_train_folder, '*.jpg')))}")
        print(f"Total processed images in {output_val_folder}: {len(glob(os.path.join(output_val_folder, '*.jpg')))}")

def process_test_data(input_root, output_root):
    """
    Process test data separately, maintaining category structure.
    
    Args:
        input_root: Root directory of the original dataset
        output_root: Root directory to save processed images
    """
    categories = ['Benign', 'Malignant']
    
    for category in categories:
        test_input_folder = os.path.join(input_root, 'test', category)
        test_output_folder = os.path.join(output_root, 'test', category) 
        
        if not os.path.exists(test_input_folder):
            print(f"Warning: Test folder {test_input_folder} not found, skipping.")
            continue       
        os.makedirs(test_output_folder, exist_ok=True)        
        
        test_image_paths = glob(os.path.join(test_input_folder, '*.jpg'))       
        print(f"Found {len(test_image_paths)} test images in {test_input_folder}")
        
       
        for img_path in test_image_paths:
            img_processed = preprocessing(img_path)
            output_path = os.path.join(test_output_folder, os.path.basename(img_path))
            cv2.imwrite(output_path, img_processed)
            print(f"Saved processed test image to {output_path}")
        
        print(f"Total processed images in {test_output_folder}: {len(glob(os.path.join(test_output_folder, '*.jpg')))}")

#
def main():
    parser = argparse.ArgumentParser(description="Process and split skin cancer dataset.")
    parser.add_argument("--input_root", type=str, required=True, help="Path to input dataset directory.")
    parser.add_argument("--output_root", type=str, required=True, help="Path to output processed dataset directory.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio (default: 0.15).")
    
    args = parser.parse_args()

    process_and_split_data(args.input_root, args.output_root, val_split=args.val_split)
    process_test_data(args.input_root, args.output_root)

if __name__ == "__main__":
    main()
