import os
import random
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set random seed for reproducibility
np.random.seed(123)

# Paths for the input and output folders
root_folder = '/home/lalitha/17_PARSeq_Github/PARSeq_Indic_HTR/basic_data/hindi/test/'  # Edit: Change file names here
input_imgs_folder = os.path.join(root_folder, 'images')
input_gt_file_path = os.path.join(root_folder, 'gt_file.txt')

output_root = '/home/lalitha/17_PARSeq_Github/PARSeq_Indic_HTR/augment_data/hindi/test/'  # Edit: Change file names here
output_imgs_folder = os.path.join(output_root, 'images')
output_gt_file_path = os.path.join(output_root, 'gt_file.txt')

# Ensure the output directory exists
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_imgs_folder, exist_ok=True)

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
        parts = line.strip().split(' ')
        if len(parts) == 2:
            image_name, label = parts
            image_labels[image_name] = label

def process_image(filename):
    try:
        # Read and augment the image
        img_path = os.path.join(input_imgs_folder, filename)
        img = Image.open(img_path)
        original_img = img.copy()
        label = image_labels.get(filename, "Unknown")
        entries = []

        # Augmentation 1: Blur
        blur_radius = random.randint(0, 5)
        if blur_radius != 0:
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        output_filename = f"blurred_{filename}"
        img.save(os.path.join(output_imgs_folder, output_filename))
        entries.append(f"{output_filename} {label}")


        # Augmentation 2: Pad/Crop
        img = original_img.copy()
        rand_num = random.randint(-50, 50)
        if rand_num < 0:
            crop_pixels = abs(rand_num)
            img = img.crop((crop_pixels, crop_pixels, img.width - crop_pixels, img.height - crop_pixels))
        elif rand_num > 0:
            avg_color = get_boundary_average_color(img)
            img = ImageOps.expand(img, border=rand_num, fill=avg_color)
        output_filename = f"augmented_{filename}"
        img.save(os.path.join(output_imgs_folder, output_filename))
        entries.append(f"{output_filename} {label}")

        # Augmentation 3: Rotate
        img = original_img.copy()
        rotation_angle = random.uniform(-30, 30)
        avg_color = get_boundary_average_color(img)
        img = img.rotate(rotation_angle, expand=True, fillcolor=avg_color)
        img = img.resize((original_img.width, original_img.height), Image.Resampling.LANCZOS)
        output_filename = f"rotated_{filename}"
        img.save(os.path.join(output_imgs_folder, output_filename))
        entries.append(f"{output_filename} {label}")

        return entries
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

# Main processing loop with ThreadPoolExecutor
image_files = [f for f in os.listdir(input_imgs_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your CPU
    all_entries = list(tqdm(executor.map(process_image, image_files), total=len(image_files)))

# Write all labels at once
with open(output_gt_file_path, 'w') as label_file:
    for entries in all_entries:
        label_file.writelines("\n".join(entries) + "\n")

print("Augmentation completed for all images in the folder.")
