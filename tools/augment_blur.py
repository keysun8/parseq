import os
import random
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm 

# Set random seed for reproducibility
np.random.seed(123)

# Paths for the input and output folders
root_folder = '../data/hindi_data/'  # edit: change file names here 
input_imgs_folder = os.path.join(root_folder, 'images')
input_gt_file_path = os.path.join(root_folder, 'gt_file.txt')

output_root = '../output/hindi_data/blur/'  # edit: change file names here 
output_imgs_folder = os.path.join(output_root, 'images')
output_gt_file_path = os.path.join(output_root, 'gt_file.txt')

# Ensure the output directory exists
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_imgs_folder, exist_ok=True)

# Load ground truth labels from gt_file.txt
image_labels = {}
with open(input_gt_file_path, 'r') as gt_file:
    for line in gt_file:
        parts = line.strip().split(' ')  # Split by space
        if len(parts) == 2:
            image_name, label = parts
            image_labels[image_name] = label

# Process each image in the input folder
with open(output_gt_file_path, 'w') as label_file:
    for filename in tqdm(os.listdir(input_imgs_folder), desc='augmenting images'):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # Specify image formats you want to process
            img_path = os.path.join(input_imgs_folder, filename)
            img = Image.open(img_path)
            original_img = img.copy()

            # Generate a random blur radius between 1 and 5
            blur_radius = random.randint(0, 5)
            # print(f'random number {blur_radius}')

            # Apply Gaussian blur to the image
            if blur_radius == 0:
                continue
            else: 
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))

            # Save the augmented image to the output folder
            output_filename = f"blurred_{filename}"
            output_path = os.path.join(output_imgs_folder, output_filename)
            img.save(output_path)

            # Retrieve label from ground truth and save as a text file in the required format
            label = image_labels.get(filename, "Unknown")  # Default to "Unknown" if not found
            label_entry = f"{output_filename} {label}\n"  # Use space as the separator
            label_file.write(label_entry)

print("Augmentation completed for all images in the folder.")
