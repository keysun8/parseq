import os
import random
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
# Set random seed for reproducibility
np.random.seed(123)

# Paths for the input and output folders
root_folder = '../data/hindi_data/' #edit: change file names here 
input_imgs_folder = os.path.join(root_folder, 'images')
input_gt_file_path = os.path.join(root_folder, 'gt_file.txt')

output_root = '../output/hindi_data/pad/' #edit: change file names here 
output_imgs_folder = os.path.join(output_root, 'images')
output_gt_file_path = os.path.join(output_root, 'gt_file.txt')

# Ensure the output directory exists
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_imgs_folder, exist_ok=True)

# Function to calculate the average boundary color
def get_boundary_average_color_old(img):
    img_array = np.array(img)
    top_edge = img_array[0, :, :]
    bottom_edge = img_array[-1, :, :]
    left_edge = img_array[:, 0, :]
    right_edge = img_array[:, -1, :]
    boundary_pixels = np.concatenate((top_edge, bottom_edge, left_edge, right_edge), axis=0)
    avg_color = boundary_pixels.mean(axis=0).astype(int)
    return tuple(avg_color)

def get_boundary_average_color(img):
    img_array = np.array(img)
    height, width, _ = img_array.shape

    # Calculate 10% of height and width
    h_margin = max(1, height // 10)  # At least 1 pixel to avoid zero margins
    w_margin = max(1, width // 10)

    # Extract 10% edge regions
    top_edge = img_array[:h_margin, :, :]
    bottom_edge = img_array[-h_margin:, :, :]
    left_edge = img_array[:, :w_margin, :]
    right_edge = img_array[:, -w_margin:, :]

    # Combine all edge pixels
    boundary_pixels = np.concatenate((top_edge.reshape(-1, 3), 
                                       bottom_edge.reshape(-1, 3), 
                                       left_edge.reshape(-1, 3), 
                                       right_edge.reshape(-1, 3)), axis=0)
    
    # Calculate average color
    avg_color = boundary_pixels.mean(axis=0).astype(int)
    return tuple(avg_color)


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

            # Generate a random number between -10 and 10 for augmentation
            rand_num = random.randint(-50, 50)
            # print(f"Processing {filename} with random number: {rand_num}")

            # Augmentation logic based on random number
            if rand_num < 0:
                # Crop image by |rand_num| pixels on each side
                crop_pixels = abs(rand_num)
                width, height = img.size
                img = img.crop((crop_pixels, crop_pixels, width - crop_pixels, height - crop_pixels))
            elif rand_num > 0:
                # Pad image by rand_num pixels on each side with the average boundary color
                avg_color = get_boundary_average_color(img)
                img = ImageOps.expand(img, border=rand_num, fill=avg_color)
            # Else, no augmentation is performed

            # Save the augmented image to the output folder
            output_filename = f"augmented_{filename}"
            output_path = os.path.join(output_imgs_folder, output_filename)
            img.save(output_path)

            # Retrieve label from ground truth and save as a text file in the required format
            label = image_labels.get(filename, "Unknown")  # Default to "Unknown" if not found
            label_entry = f"{output_filename} {label}\n"  # Use space as the separator
            label_file.write(label_entry)

print("Augmentation completed for all images in the folder.")
