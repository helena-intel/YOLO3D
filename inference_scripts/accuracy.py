"""Compare native PyTorch and OpenVINO YOLO/ResNet models on labeled images.

The local KITTI folder format and Hugging Face ``nateraw/kitti`` are the
datasets tested by this script. Other datasets need code to map their image
and annotation formats to the fields used here; matching column names alone
does not guarantee compatible labels or class IDs.

YOLO evaluation measures 2D detection quality for KITTI Car and Pedestrian
labels. ResNet evaluation uses labeled 2D boxes as ground-truth crops, so it
measures the 3D regression model separately from YOLO detection quality. The
reported numbers are comparison metrics, not a general-purpose model score.

Examples:
    python accuracy.py yolo --source-model weights/yolov5s.pt --openvino-model weights/openvino/yolov5s.xml --hf-dataset nateraw/kitti --device CPU
    python accuracy.py resnet --source-model weights/resnet18.pkl --openvino-model weights/openvino/resnet18.xml --hf-dataset nateraw/kitti --device NPU --batch-size 8
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for logger_name in ("httpx", "httpx2", "httpcore"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

from library.Calib import get_P
from script.ClassAverages import ClassAverages
from script.Dataset import generate_bins
from script.Model import ResNet18
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.metrics import ap_per_class, box_iou
from models.common import DetectMultiBackend


# These KITTI labels have an unambiguous equivalent in the COCO model classes.
# Other KITTI categories are excluded rather than treating a different object
# category as ground truth for a COCO prediction.
KITTI_TO_COCO = {"Car": 2, "Pedestrian": 0}
COCO_NAMES = {0: "person", 2: "car"}


def parse_kitti_labels(label_path):
    """Read KITTI labels needed by the detector and 3D regressor evaluators."""
    labels = []
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 15 or fields[0] == "DontCare":
            continue
        labels.append(
            {
                "class_name": fields[0],
                "alpha": float(fields[3]),
                "box": np.asarray(fields[4:8], dtype=np.float32),
                "dimensions": np.asarray(fields[8:11], dtype=np.float32),
            }
        )
    return labels


def iter_local_samples(dataset_dir, limit):
    """Yield images and parsed labels from a local KITTI split.

    Expected layout: ``image_2/<frame>.<ext>`` and ``label_2/<frame>.txt``.
    Other local dataset layouts need an adapter.
    """
    image_dir = dataset_dir / "image_2"
    label_dir = dataset_dir / "label_2"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected KITTI image_2 and label_2 directories under {dataset_dir}. "
            "Provide a downloaded labeled KITTI training or validation split."
        )

    image_paths = sorted(
        path for extension in ("*.png", "*.jpg", "*.jpeg") for path in image_dir.glob(extension)
    )
    count = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise RuntimeError(f"Could not read {image_path}")
            yield image_bgr, parse_kitti_labels(label_path)
            count += 1
            if limit and count >= limit:
                return


def parse_huggingface_labels(labels):
    """Convert nateraw/kitti labels to the evaluator's internal format.

    This adapter expects KITTI-style ``type``, ``bbox``, ``alpha``, and
    ``dimensions`` fields. Other annotation schemas need a mapping for their
    fields and class names before using these evaluation functions.
    """
    return [
        {
            "class_name": label["type"],
            "alpha": float(label["alpha"]),
            "box": np.asarray(label["bbox"], dtype=np.float32),
            "dimensions": np.asarray(label["dimensions"], dtype=np.float32),
        }
        for label in labels
        if label["type"] != "DontCare"
    ]


def iter_huggingface_samples(args):
    """Stream a bounded sample from a KITTI-compatible Hugging Face dataset.

    Streaming limits local storage, but requested images are still
    transferred. The dataset must provide the configured image column and
    labels understood by :func:`parse_huggingface_labels`.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Install Hugging Face datasets with: pip install datasets") from error

    dataset = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    if args.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
    for count, sample in enumerate(dataset, start=1):
        image_rgb = np.asarray(sample[args.hf_image_column].convert("RGB"))
        yield cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), parse_huggingface_labels(sample[args.hf_label_column])
        if args.limit and count >= args.limit:
            return


def iter_samples(args):
    """Yield evaluation samples from either local KITTI files or Hugging Face."""
    if args.hf_dataset:
        yield from iter_huggingface_samples(args)
    else:
        yield from iter_local_samples(args.dataset, args.limit)


def preprocess_yolo_uint8(image_bgr, image_size):
    resized = letterbox(image_bgr, new_shape=image_size, stride=32, auto=False)[0]
    bgr_chw = resized.transpose(2, 0, 1)
    return np.ascontiguousarray(bgr_chw, dtype=np.uint8)[None]


def preprocess_yolo(image_bgr, image_size):
    bgr_chw = preprocess_yolo_uint8(image_bgr, image_size)[0]
    rgb_chw = bgr_chw[::-1]
    return np.ascontiguousarray(rgb_chw, dtype=np.float32)[None] / 255.0


def correct_detections(detections, labels, iou_thresholds):
    """Match predicted boxes to labels using intersection-over-union thresholds.

    A prediction is correct when its class matches and enough of its area
    overlaps a ground-truth box. Higher thresholds require tighter alignment
    and contribute to the reported mAP@0.5:0.95 metric.
    """
    correct = torch.zeros((detections.shape[0], len(iou_thresholds)), dtype=torch.bool)
    if not len(labels) or not len(detections):
        return correct

    iou = box_iou(labels[:, 1:], detections[:, :4])
    matches = torch.where((iou >= iou_thresholds[0]) & (labels[:, 0:1] == detections[:, 5]))
    if matches[0].numel() == 0:
        return correct

    match_data = torch.cat((torch.stack(matches, 1), iou[matches[0], matches[1]][:, None]), 1).cpu().numpy()
    if len(match_data) > 1:
        match_data = match_data[match_data[:, 2].argsort()[::-1]]
        match_data = match_data[np.unique(match_data[:, 1], return_index=True)[1]]
        match_data = match_data[match_data[:, 2].argsort()[::-1]]
        match_data = match_data[np.unique(match_data[:, 0], return_index=True)[1]]
    correct[match_data[:, 1].astype(int)] = torch.tensor(match_data[:, 2:3]) >= iou_thresholds
    return correct


class YoloPredictor:
    """Run a YOLOv5 source model on CPU or an OpenVINO IR on the requested device."""

    def __init__(self, model_path, backend, openvino_device):
        self.backend = backend
        if backend == "source":
            self.model = DetectMultiBackend(str(model_path), device=torch.device("cpu"), data=ROOT / "data/coco.yaml")
            self.output = None
        else:
            core = ov.Core()
            model = core.read_model(str(model_path))
            if model.input(0).partial_shape.is_dynamic:
                model.reshape({model.input(0).any_name: ov.PartialShape([1, 3, 640, 640])})
            self.model = core.compile_model(model, openvino_device)
            self.output = self.model.output(0)

    def predict(self, image_bgr, image_size, conf_threshold, iou_threshold):
        input_data = preprocess_yolo(image_bgr, image_size)
        if self.backend == "source":
            with torch.inference_mode():
                prediction = self.model(torch.from_numpy(input_data))
        else:
            prediction = torch.from_numpy(self.model(input_data)[self.output])
        detections = non_max_suppression(
            prediction,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=sorted(COCO_NAMES),
        )[0]
        if len(detections):
            detections[:, :4] = scale_coords(input_data.shape[2:], detections[:, :4], image_bgr.shape).round()
        return detections.cpu()


def evaluate_yolo(predictor, args):
    """Return COCO-style mAP statistics for a YOLO predictor."""
    iou_thresholds = torch.linspace(0.5, 0.95, 10)
    statistics = []
    images = 0
    for image_bgr, kitti_labels in iter_samples(args):
        labels = [label for label in kitti_labels if label["class_name"] in KITTI_TO_COCO]
        target = torch.tensor(
            [[KITTI_TO_COCO[label["class_name"]], *label["box"]] for label in labels], dtype=torch.float32
        ).reshape(-1, 5)
        detections = predictor.predict(image_bgr, args.image_size, args.conf_threshold, args.iou_threshold)
        statistics.append((correct_detections(detections, target, iou_thresholds), detections[:, 4], detections[:, 5], target[:, 0]))
        images += 1

    if not images:
        raise RuntimeError("No paired KITTI images and labels were found.")
    correct, confidence, predicted_classes, target_classes = [torch.cat(values, 0).cpu().numpy() for values in zip(*statistics)]
    if not len(target_classes):
        raise RuntimeError("The selected split contains no Car or Pedestrian labels to evaluate.")
    _, _, precision, recall, _, average_precision, classes = ap_per_class(
        correct, confidence, predicted_classes, target_classes, names=COCO_NAMES
    )
    return {
        "images": images,
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "map50": float(average_precision[:, 0].mean()),
        "map5095": float(average_precision.mean()),
        "classes": [COCO_NAMES[int(class_id)] for class_id in classes],
    }


def preprocess_resnet_uint8(image_bgr, box):
    """Create the raw uint8 BGR crop accepted by the OpenVINO preprocessing graph."""
    x1, y1, x2, y2 = np.rint(box).astype(int)
    crop = image_bgr[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1]
    if crop.size == 0:
        raise ValueError(f"Invalid crop box: {box.tolist()}")
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.uint8)


def preprocess_resnet(image_bgr, box):
    """Match the BGR crop, ToTensor, and ImageNet normalization used in training."""
    uint8_input = preprocess_resnet_uint8(image_bgr, box)
    chw = uint8_input[0].astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return ((chw - mean) / std)[None]


def decode_resnet_outputs(outputs, averages, class_name, angle_bins, class_dimensions=None):
    """Convert network heads into absolute dimensions and alpha orientation."""
    orientation = next(output for output in outputs if output.ndim == 3)[0]
    confidence = next(output for output in outputs if output.ndim == 2 and output.shape[1] == 2)[0]
    dimension = next(output for output in outputs if output.ndim == 2 and output.shape[1] == 3)[0]
    bin_index = int(np.argmax(confidence))
    alpha = np.arctan2(orientation[bin_index, 1], orientation[bin_index, 0]) + angle_bins[bin_index] - np.pi
    if class_dimensions is None:
        class_dimensions = averages.get_item(class_name)
    return alpha, dimension + class_dimensions


class ResNetPredictor:
    """Run the converted ResNet18 source model or an OpenVINO IR."""

    def __init__(self, model_path, backend, openvino_device):
        self.backend = backend
        if backend == "source":
            model = ResNet18(model=resnet18(weights=None, progress=False))
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.model = model.eval()
        else:
            core = ov.Core()
            model = core.read_model(str(model_path))
            self.model = core.compile_model(model, openvino_device)
            self.request = self.model.create_infer_request()
            self.input_partial_shape = self.model.input(0).partial_shape

    def preprocess(self, image_bgr, box):
        if self.backend == "openvino":
            return preprocess_resnet_uint8(image_bgr, box)
        return preprocess_resnet(image_bgr, box)

    def predict(self, input_data):
        if self.backend == "source":
            with torch.inference_mode():
                return [output.cpu().numpy() for output in self.model(torch.from_numpy(input_data))]
        if not self.input_partial_shape.compatible(ov.PartialShape(input_data.shape)):
            raise ValueError(
                f"OpenVINO model input shape {self.input_partial_shape} does not support "
                f"input shape {tuple(input_data.shape)}."
            )
        self.request.set_input_tensor(ov.Tensor(input_data))
        self.request.infer()
        return [self.request.get_output_tensor(index).data.copy() for index in range(len(self.model.outputs))]


def wrapped_angle_error_degrees(predicted, target):
    difference = np.arctan2(np.sin(predicted - target), np.cos(predicted - target))
    return abs(np.degrees(difference))


def evaluate_resnet(predictor, args):
    """Evaluate 3D regression using labeled boxes as object crops.

    Ground-truth boxes bypass YOLO so these metrics isolate ResNet's
    orientation and dimension predictions. Classes without learned average
    dimensions are skipped because the model predicts residuals that require
    those averages to recover real object dimensions.
    """
    averages = ClassAverages()
    angle_bins = generate_bins(2)
    angle_errors = []
    dimension_errors = []
    objects = 0
    skipped_objects = 0
    images = 0

    def evaluate_batch(batch):
        nonlocal objects
        input_data = np.concatenate([item[0] for item in batch], axis=0)
        outputs = predictor.predict(input_data)
        output_batches = {output.shape[0] for output in outputs}
        if output_batches != {len(batch)}:
            raise RuntimeError(
                f"Predictor returned output batch sizes {sorted(output_batches)} "
                f"for input batch {len(batch)}."
            )
        for index, (_, label, class_dimensions) in enumerate(batch):
            alpha, dimensions = decode_resnet_outputs(
                [output[index:index + 1] for output in outputs],
                averages,
                label["class_name"],
                angle_bins,
                class_dimensions,
            )
            angle_errors.append(wrapped_angle_error_degrees(alpha, label["alpha"]))
            dimension_errors.append(np.abs(dimensions - label["dimensions"]))
            objects += 1

    batch = []
    for image_bgr, labels in iter_samples(args):
        images += 1
        for label in labels:
            if not averages.recognized_class(label["class_name"]):
                continue
            class_average = averages.dimension_map[label["class_name"].lower()]
            if class_average["count"] == 0:
                skipped_objects += 1
                continue
            class_dimensions = averages.get_item(label["class_name"])
            if not np.isfinite(class_dimensions).all():
                skipped_objects += 1
                continue
            batch.append((predictor.preprocess(image_bgr, label["box"]), label, class_dimensions))
            if len(batch) == args.batch_size:
                evaluate_batch(batch)
                batch = []

    if batch:
        evaluate_batch(batch)

    if not objects:
        raise RuntimeError("No labels match the classes in script/class_averages.txt.")
    dimension_errors = np.asarray(dimension_errors)
    return {
        "images": images,
        "objects": objects,
        "skipped_objects": skipped_objects,
        "orientation_mae_degrees": float(np.mean(angle_errors)),
        "dimensions_mae": dimension_errors.mean(axis=0),
        "mean_dimension_mae": float(dimension_errors.mean()),
    }


def print_yolo_metrics(name, metrics):
    print(f"\n{name} YOLO accuracy")
    print(f"  Images: {metrics['images']} | evaluated classes: {', '.join(metrics['classes'])}")
    print(f"  Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    print(f"  mAP@0.5: {metrics['map50']:.4f} | mAP@0.5:0.95: {metrics['map5095']:.4f}")


def print_resnet_metrics(name, metrics):
    height, width, length = metrics["dimensions_mae"]
    print(f"\n{name} ResNet accuracy (ground-truth 2D crops)")
    print(
        f"  Images: {metrics['images']} | objects: {metrics['objects']} "
        f"| skipped without dimension averages: {metrics['skipped_objects']}"
    )
    print(f"  Orientation MAE: {metrics['orientation_mae_degrees']:.3f} degrees")
    print(f"  Dimension MAE [height, width, length]: [{height:.4f}, {width:.4f}, {length:.4f}] m")
    print(f"  Mean dimension MAE: {metrics['mean_dimension_mae']:.4f} m")


def add_common_arguments(parser):
    parser.add_argument("--source-model", type=Path, help="Native PyTorch model (.pt for YOLO, .pkl for ResNet).")
    parser.add_argument("--openvino-model", type=Path, help="OpenVINO IR XML model to compare.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device, for example CPU, GPU, or NPU.")
    dataset_source = parser.add_mutually_exclusive_group()
    dataset_source.add_argument("--dataset", type=Path, default=ROOT / "dataset/KITTI/training", help="Local KITTI split containing image_2 and label_2.")
    dataset_source.add_argument(
        "--hf-dataset",
        help="Hugging Face dataset ID (tested with nateraw/kitti; other schemas need an adapter).",
    )
    parser.add_argument("--hf-split", default="train", help="Hugging Face split to stream.")
    parser.add_argument("--hf-image-column", default="image", help="Image column in the Hugging Face dataset.")
    parser.add_argument("--hf-label-column", default="label", help="KITTI label-list column in the Hugging Face dataset.")
    parser.add_argument("--shuffle-seed", type=int, help="Seed for a reproducible shuffled Hugging Face sample; omit to preserve source order.")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000, help="Streaming shuffle buffer size for Hugging Face data.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum images to evaluate; 0 evaluates all images.")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="task", required=True)
    yolo_parser = subparsers.add_parser("yolo", help="Evaluate 2D detection mAP.")
    add_common_arguments(yolo_parser)
    yolo_parser.add_argument("--image-size", type=int, default=640)
    yolo_parser.add_argument("--conf-threshold", type=float, default=0.001, help="Low threshold required for AP evaluation.")
    yolo_parser.add_argument("--iou-threshold", type=float, default=0.6, help="NMS IoU threshold.")
    resnet_parser = subparsers.add_parser("resnet", help="Evaluate 3D attribute regression.")
    add_common_arguments(resnet_parser)
    resnet_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of ResNet crops per inference request. Dynamic OpenVINO IRs support variable final batches.",
    )
    args = parser.parse_args()
    if not args.source_model and not args.openvino_model:
        parser.error("provide --source-model, --openvino-model, or both")
    for model_path in (args.source_model, args.openvino_model):
        if model_path and not model_path.is_file():
            parser.error(f"model file does not exist: {model_path}")
    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer")
    if args.task == "resnet" and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    return args


def main():
    args = parse_arguments()
    if args.task == "yolo":
        if args.source_model:
            print_yolo_metrics("PyTorch", evaluate_yolo(YoloPredictor(args.source_model, "source", args.device), args))
        if args.openvino_model:
            print_yolo_metrics("OpenVINO", evaluate_yolo(YoloPredictor(args.openvino_model, "openvino", args.device), args))
    else:
        if args.source_model:
            print_resnet_metrics("PyTorch", evaluate_resnet(ResNetPredictor(args.source_model, "source", args.device), args))
        if args.openvino_model:
            print_resnet_metrics("OpenVINO", evaluate_resnet(ResNetPredictor(args.openvino_model, "openvino", args.device), args))


if __name__ == "__main__":
    main()