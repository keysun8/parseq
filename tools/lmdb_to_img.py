import os
import lmdb
import numpy as np
from PIL import Image
import argparse
import io
from tqdm import tqdm

def convertLMDBToImagesAndLabels(lmdbPath, outputImagePath, outputLabelPath):
    """
    Convert LMDB dataset back to images and labels.
    ARGS:
        lmdbPath        : path to the LMDB dataset
        outputImagePath : directory to save images
        outputLabelPath : file to save labels
    """
    # Create output directories if they do not exist
    os.makedirs(outputImagePath, exist_ok=True)

    # Open the LMDB environment
    env = lmdb.open(lmdbPath, readonly=True)
    
    # Prepare to write labels to a file
    with open(outputLabelPath, 'w', encoding='utf-8') as labelFile:
        with env.begin() as txn:
            # Read the number of samples
            num_samples = int(txn.get('num-samples'.encode()).decode())

            # for i in tqdm(range(1, num_samples + 1), desc='converting to images'):
            for i in tqdm(range(num_samples), desc='converting to images'):
            # for i in tqdm(range(10), desc='converting to images'):
                imageKey = 'image-%09d'.encode() % i
                labelKey = 'label-%09d'.encode() % i
                
                # Retrieve image and label from LMDB
                imageBin = txn.get(imageKey)
                label = txn.get(labelKey)

                # Check if both image and label exist
                if imageBin is not None and label is not None:
                    # Save the image
                    image = Image.open(io.BytesIO(imageBin))
                    image.save(os.path.join(outputImagePath, f'{i:09d}.png'))

                    # Write the label to the file
                    labelFile.write(f'{i:09d}.png {label.decode()}\n')

    env.close()
    print(f'Converted LMDB dataset to images in {outputImagePath} and labels in {outputLabelPath}')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Convert LMDB to Images and Labels")
    # parser.add_argument("--lmdbPath", type=str, help="path to the LMDB dataset")
    # parser.add_argument("--outputImagePath", type=str, help="directory to save images")
    # parser.add_argument("--outputLabelPath", type=str, help="file to save labels")

    # args = parser.parse_args()
    # lmdbPath = args.lmdbPath
    # outputImagePath = args.outputImagePath
    # outputLabelPath = args.outputLabelPath
    lmdbPath = '../../../7_handwritten_dataset/hindi/train_lmdb'
    outputImagePath = '/home/lalitha/17_PARSeq_Github/PARSeq_Indic_HTR/basic_data/hindi/train/images/'
    outputLabelPath = '/home/lalitha/17_PARSeq_Github/PARSeq_Indic_HTR/basic_data/hindi/train/gt_file.txt'

    convertLMDBToImagesAndLabels(lmdbPath, outputImagePath, outputLabelPath)

'''
python lmdb_to_img.py --lmdbPath /home/lalitha/17_PARSeq_Github/PARSeq_Indic_HTR/data --outputImagePath ../data/hindi_data/images/ --outputLabelPath ../data/hindi_data/gt_file.txt
'''