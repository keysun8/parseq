#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as T

from strhub.models.utils import load_from_checkpoint, parse_model_args


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint path")
    parser.add_argument('--image_dir', required=True, help="Folder containing images")
    parser.add_argument('--output_file', default='predictions.txt')
    parser.add_argument('--device', default='cuda')

    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    # Load model
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    # Image transform (same as datamodule)
    transform = T.Compose([
        T.Resize(hp.img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])

    image_paths = sorted([
        p for p in Path(args.image_dir).glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    if len(image_paths) == 0:
        print("No images found in folder:", args.image_dir)
        return

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("image_name\tprediction\tconfidence\n")

        for img_path in tqdm(image_paths, desc="Predicting"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Could not read {img_path}: {e}")
                continue

            img_tensor = transform(img).unsqueeze(0).to(args.device)

            # Forward pass
            probs = model(img_tensor).softmax(-1)
            pred, prob = model.tokenizer.decode(probs)

            text = model.charset_adapter(pred[0])
            confidence = prob[0].prod().item()

            f.write(f"{img_path.name}\t{text}\t{confidence:.4f}\n")

    print(f"\n Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
