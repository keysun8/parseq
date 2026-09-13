import os
import lmdb
import numpy as np
from PIL import Image
import argparse
import io
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_image_and_label(index, txn, outputImagePath):
    """
    Save image and label from LMDB transaction.
    """
    imageKey = f'image-{index:09d}'.encode()
    labelKey = f'label-{index:09d}'.encode()

    imageBin = txn.get(imageKey)
    label = txn.get(labelKey)

    if imageBin is not None and label is not None:
        # Save the image
        image = Image.open(io.BytesIO(imageBin))
        image.save(os.path.join(outputImagePath, f'{index:09d}.png'))

        # Return the label entry
        return f'{index:09d}.png {label.decode()}'
    return None

def convertLMDBToImagesAndLabels(lmdbPath, outputImagePath, outputLabelPath, num_workers=4):
    """
    Convert LMDB dataset back to images and labels.
    ARGS:
        lmdbPath        : path to the LMDB dataset
        outputImagePath : directory to save images
        outputLabelPath : file to save labels
        num_workers     : number of threads to use for processing
    """
    os.makedirs(outputImagePath, exist_ok=True)

    # Open the LMDB environment
    env = lmdb.open(lmdbPath, readonly=True)

    with env.begin() as txn:
        num_samples = int(txn.get('num-samples'.encode()).decode())

        # Prepare to write labels to a file
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            label_entries = list(tqdm(
                executor.map(lambda i: save_image_and_label(i, txn, outputImagePath), range(num_samples)),
                total=num_samples,
                desc='Converting LMDB to images'
            ))

    # Write the labels to the output file
    with open(outputLabelPath, 'w', encoding='utf-8') as labelFile:
        labelFile.writelines(entry + '\n' for entry in label_entries if entry)

    env.close()
    print(f'Converted LMDB dataset to images in {outputImagePath} and labels in {outputLabelPath}')

if __name__ == '__main__':
    lmdbPath = '/home/lalitha/7_handwritten_dataset/hindi/test_lmdb/'
    outputImagePath = '/home/lalitha/17_PARSeq_Github/PARSeq_Indic_HTR/basic_data/hindi/test/images/'
    outputLabelPath = '/home/lalitha/17_PARSeq_Github/PARSeq_Indic_HTR/basic_data/hindi/test/gt_file.txt'

    convertLMDBToImagesAndLabels(lmdbPath, outputImagePath, outputLabelPath, num_workers=8)

'''
python lmdb_to_img.py --lmdbPath /home/lalitha/17_PARSeq_Github/PARSeq_Indic_HTR/data --outputImagePath ../data/hindi_data/images/ --outputLabelPath ../data/hindi_data/gt_file.txt
'''
