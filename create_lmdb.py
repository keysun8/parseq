import io
import os
import json

import lmdb
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
word_image_dir = "/home/kishan/diffusion/Syn_Data/syn_hw_merged_words"  # folder containing the word-level crops
word_gt_json = "/home/kishan/diffusion/Syn_Data/syn_hw_merged_words.json"
gt_txt_path = "/home/kishan/diffusion/Syn_Data/syn_hw_merged_words_gt.txt"
lmdb_output_root = "/home/kishan/diffusion/Syn_Data/lmdb"
lmdb_output_name = "syn_hw_merged_words"  


def build_gt_txt(word_gt_json_path, gt_txt_out_path):
    """
    Convert {crop_name: label} JSON into CRNN-style gt file lines:
        <relative_image_path> <label>
    Skips entries whose label contains whitespace (would break the
    maxsplit=1 parsing downstream) and logs them.
    """
    with open(word_gt_json_path, 'r', encoding='utf-8') as f:
        word_gt = json.load(f)

    skipped = 0
    with open(gt_txt_out_path, 'w', encoding='utf-8') as f:
        for crop_name, label in word_gt.items():
            if any(ch.isspace() for ch in label):
                print(f"Skipping '{crop_name}': label contains whitespace ('{label}')")
                skipped += 1
                continue
            f.write(f"{crop_name} {label}\n")

    print(f"gt file written: {gt_txt_out_path} ({len(word_gt) - skipped} entries, {skipped} skipped)")
    return gt_txt_out_path


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    img = Image.open(io.BytesIO(imageBin)).convert('RGB')
    return np.prod(img.size) > 0


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image

    Any GT line that can't be turned into a valid (image, label) pair is
    SKIPPED (not fatal) and logged to <outputPath>/error_image_log.txt.
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1
    n_skipped = 0
    error_log_path = os.path.join(outputPath, 'error_image_log.txt')

    def log_skip(i, reason):
        nonlocal n_skipped
        n_skipped += 1
        with open(error_log_path, 'a', encoding='utf-8') as log:
            log.write(f'{i}-th line skipped: {reason}\n')

    with open(gtFile, 'r', encoding='utf-8') as f:
        data = f.readlines()

    nSamples = len(data)
    for i, line in enumerate(data):
        line = line.strip()
        if not line:
            log_skip(i, 'empty line')
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            log_skip(i, f'could not split into imagePath/label: {line!r}')
            continue
        imagePath, label = parts
        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            log_skip(i, f'image file not found: {imagePath}')
            continue

        try:
            with open(imagePath, 'rb') as f:
                imageBin = f.read()
        except OSError as e:
            log_skip(i, f'could not read image file {imagePath}: {e}')
            continue

        if checkValid:
            try:
                img = Image.open(io.BytesIO(imageBin)).convert('RGB')
            except Exception as e:
                log_skip(i, f'invalid/corrupt image {imagePath}: {e}')
                continue
            if np.prod(img.size) == 0:
                log_skip(i, f'zero-size image: {imagePath}')
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()

    if n_skipped:
        print(f'[WARN] Skipped {n_skipped} GT lines, see {error_log_path}')
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    gt_txt = build_gt_txt(word_gt_json, gt_txt_path)
    output_path = os.path.join(lmdb_output_root, lmdb_output_name)
    createDataset(
        inputPath=word_image_dir,
        gtFile=gt_txt,
        outputPath=output_path,
        checkValid=True,
    )
