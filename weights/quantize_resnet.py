#!/usr/bin/env python3
"""Quantize the ResNet OpenVINO IR with representative KITTI object crops.

Example:
    python inference_scripts/quantize_resnet.py --max-samples 300

NNCF calibration uses input tensors to measure typical activation ranges; it
does not consume labels or optimize model weights. This ResNet expects
224x224 object crops rather than full camera frames, so KITTI labels are used
only to construct representative crops. The local KITTI format and Hugging
Face nateraw/kitti are the datasets tested with this script. Other datasets
need an adapter for their image/box schema and compatible crop preprocessing.
The official nateraw/kitti test split has no labels; calibrate from the tail
of its labeled train split, or supply crops obtained from a detector.

INT8 uses 8-bit values and can reduce model size or improve inference speed,
but may slightly change predictions. Compare against the original model on
separate evaluation images after calibration.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for logger_name in ("httpx", "httpx2", "httpcore"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantize a ResNet IR with representative KITTI object crops.")
    parser.add_argument("--ir", type=Path, default=ROOT / "weights/openvino/resnet18.xml", help="Input OpenVINO XML model.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "weights/openvino/resnet18_int8.xml",
        help="Output quantized OpenVINO XML model.",
    )
    parser.add_argument("--hf-dataset", default="nateraw/kitti", help="Hugging Face KITTI-compatible dataset ID.")
    parser.add_argument("--hf-split", default="train", help="Labeled Hugging Face split to stream.")
    parser.add_argument("--hf-image-column", default="image", help="Image column in the Hugging Face dataset.")
    parser.add_argument("--hf-label-column", default="label", help="KITTI label-list column in the Hugging Face dataset.")
    parser.add_argument(
        "--tail-images",
        type=int,
        default=1000,
        help="Approximate number of images from the end of the split to use for calibration.",
    )
    parser.add_argument("--max-samples", type=int, default=300, help="Maximum real object crops used for calibration.")
    args = parser.parse_args()
    if not args.ir.is_file():
        parser.error(f"IR model does not exist: {args.ir}")
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.tail_images <= 0:
        parser.error("--tail-images must be positive")
    if args.hf_dataset == "nateraw/kitti" and args.hf_split == "test":
        parser.error(
            "nateraw/kitti's test split has no labels, so it cannot provide ResNet object crops. "
            "Use --hf-split train; the default tail partition is separate from accuracy.py's leading evaluation sample."
        )
    return args


def preprocess_resnet_crop(image_bgr, box):
    """Match the raw uint8 BGR crop accepted by the baked ResNet preprocessing."""
    x1, y1, x2, y2 = np.rint(box).astype(int)
    crop = image_bgr[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.uint8)


def iter_calibration_batches(args):
    """Yield raw uint8 batch-1 tensors from real annotated KITTI crops.

    Labels only tell this script where objects are; NNCF receives the image
    tensors, not the labels. The baked IR normalizes these inputs internally.
    Other datasets need adapters for image decoding and box conversion.
    """
    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError as error:
        raise ImportError("Install calibration dependencies with: pip install datasets") from error

    builder = load_dataset_builder(args.hf_dataset)
    splits = builder.info.splits
    if splits is None or args.hf_split not in splits:
        raise RuntimeError(f"Could not find split metadata for {args.hf_dataset}/{args.hf_split}.")
    split_info = splits[args.hf_split]
    if split_info.num_examples is None:
        raise RuntimeError(f"Could not determine the number of examples in {args.hf_dataset}/{args.hf_split}.")

    dataset = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    shard_count = max(1, (split_info.num_examples + args.tail_images - 1) // args.tail_images)
    dataset = dataset.shard(num_shards=shard_count, index=shard_count - 1, contiguous=True)
    print(
        f"Streaming the final contiguous shard of {args.hf_dataset}/{args.hf_split} "
        f"(approximately the last {args.tail_images} of {split_info.num_examples} images)."
    )
    samples = 0
    for sample in dataset:
        image_rgb = np.asarray(sample[args.hf_image_column].convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        for label in sample[args.hf_label_column]:
            if label["type"] == "DontCare":
                continue
            input_data = preprocess_resnet_crop(image_bgr, np.asarray(label["bbox"], dtype=np.float32))
            if input_data is None:
                continue
            yield [input_data]
            samples += 1
            if samples >= args.max_samples:
                return

    if not samples:
        raise RuntimeError("The calibration dataset contained no valid annotated object crops.")


def main():
    args = parse_arguments()
    core = ov.Core()
    model = core.read_model(args.ir)

    print(f"Calibrating {args.ir} with up to {args.max_samples} real KITTI object crops from {args.hf_dataset}/{args.hf_split}.")
    quantized_model = nncf.quantize(
        model,
        calibration_dataset=nncf.Dataset(iter_calibration_batches(args)),
        subset_size=args.max_samples,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ov.serialize(quantized_model, args.output)
    print(f"INT8 model saved to {args.output}")


if __name__ == "__main__":
    main()