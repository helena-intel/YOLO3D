#!/usr/bin/env python3
"""
Quantize an OpenVINO IR (XML) model to INT8 using NNCF (post-training).

Usage example:
  python quantize_yolo.py \
    --ir weights/openvino/yolov5s.xml \
    --images data/calib_images/ \
    --output weights/openvino/yolov5s_int8.xml
    

This script reads an OpenVINO IR, builds a simple calibration dataset
from images in a folder, runs NNCF quantization, and writes the quantized IR.
The input IR expects letterboxed RGB float32 NCHW images scaled to [0, 1].
"""
import argparse
import os
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import nncf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.augmentations import letterbox


def parse_args():
    p = argparse.ArgumentParser(description="Quantize OpenVINO IR (YOLO) using NNCF")
    p.add_argument("--ir", required=True, help="Path to input OpenVINO XML model")
    p.add_argument("--images", required=True, help="Folder with calibration images")
    p.add_argument("--output", required=True, help="Path to write quantized XML model")
    p.add_argument("--input-size", type=int, nargs=2, default=(640, 640), help="Model input H W")
    p.add_argument("--batch", type=int, default=1, help="Batch size for calibration")
    p.add_argument("--max-samples", type=int, default=100, help="Max calibration samples to use")
    return p.parse_args()


def preprocess_image(path, input_size):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = letterbox(img, new_shape=input_size, stride=32, auto=False)[0]
    rgb = img[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb, dtype=np.float32) / 255.0


def make_calibration_generator(image_paths, input_size, batch, max_samples):
    def gen():
        batch_buf = []
        count = 0
        for p in image_paths:
            img = preprocess_image(p, input_size)
            batch_buf.append(img)
            count += 1
            if len(batch_buf) == batch:
                yield [np.stack(batch_buf, axis=0)]
                batch_buf = []
            if count >= max_samples:
                break
        if batch_buf:
            yield [np.stack(batch_buf, axis=0)]
    return gen


def main():
    args = parse_args()

    core = ov.Core()
    model = core.read_model(args.ir)

    # gather image paths
    imgs = sorted(glob.glob(os.path.join(args.images, "**", "*.jpg"), recursive=True))
    imgs += sorted(glob.glob(os.path.join(args.images, "**", "*.png"), recursive=True))
    if len(imgs) == 0:
        raise SystemExit("No images found in --images folder for calibration")

    calib_gen = make_calibration_generator(imgs, args.input_size, args.batch, args.max_samples)

    quantized_model = nncf.quantize(
        model,
        calibration_dataset=nncf.Dataset(calib_gen()),
        subset_size=args.max_samples,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ov.serialize(quantized_model, args.output)
    print(f"Quantized model saved to {args.output}")


if __name__ == "__main__":
    main()
