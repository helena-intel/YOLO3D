# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
import numpy
import time
import argparse
from pathlib import Path
import sys
import cv2 # Explicitly import cv2
import random # For selecting images
import glob

# Add project root to sys.path for utils.datasets
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # Project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def iter_benchmark_images(path, img_size):
    """Load image-only datasets without importing PyTorch's training utilities."""
    source = str(Path(path).resolve())
    if '*' in source:
        files = [Path(item) for item in sorted(glob.glob(source, recursive=True))]
    elif Path(source).is_dir():
        files = sorted(Path(source).glob('*.*'))
    else:
        files = [Path(source)]

    image_suffixes = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo'}
    video_suffixes = {'.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv'}
    if any(item.suffix.lower() in video_suffixes for item in files):
        from utils.datasets import LoadImages

        for entry in LoadImages(source, img_size=img_size, stride=32, auto=False):
            yield entry[1]
        return

    for image_path in files:
        if image_path.suffix.lower() not in image_suffixes:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        ratio = min(img_size / height, img_size / width)
        resized_width, resized_height = round(width * ratio), round(height * ratio)
        if (width, height) != (resized_width, resized_height):
            image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        horizontal_padding = (img_size - resized_width) / 2
        vertical_padding = (img_size - resized_height) / 2
        image = cv2.copyMakeBorder(
            image,
            round(vertical_padding - 0.1), round(vertical_padding + 0.1),
            round(horizontal_padding - 0.1), round(horizontal_padding + 0.1),
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        yield numpy.ascontiguousarray(image.transpose(2, 0, 1)[::-1])

def main():
    parser = argparse.ArgumentParser(description="OpenVINO IR Model Benchmark Script with Real Dataset")
    parser.add_argument('--model_path', type=str, required=True, help='Path to OpenVINO IR model .xml file')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to dataset directory or image file. If None, uses dummy data.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size (square) for preprocessing and model input. Should match IR model expected input size if not reshaping H,W.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference. Model must support this or have dynamic batch.')
    parser.add_argument('--static-shapes', action='store_true', help='Reshape the model to the requested batch and image size before compilation.')
    parser.add_argument('--warmup_runs', type=int, default=10, help='Number of warm-up inference runs')
    parser.add_argument('--benchmark_runs', type=int, default=50, help='Number of benchmark inference runs')
    parser.add_argument('--device', type=str, default='CPU', help="OpenVINO device (e.g., 'CPU', 'GPU').")

    args = parser.parse_args()

    model_xml_path = Path(args.model_path)
    if not model_xml_path.exists():
        print(f"Error: Model XML file not found: {model_xml_path}", file=sys.stderr)
        return
    model_bin_path = model_xml_path.with_suffix(".bin")
    if not model_bin_path.exists():
        print(f"Error: Model BIN file not found: {model_bin_path}", file=sys.stderr)
        return
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    if dataset_path and not dataset_path.exists() and not dataset_path.is_file():
        if not (dataset_path.parent.exists() and '*' in dataset_path.name):
             print(f"Error: Dataset path {dataset_path} does not exist or is not a file/glob pattern.", file=sys.stderr)
             return

    # OpenVINO Model Loading
    final_input_layer = None
    final_input_shape_tuple = None
    element_type_numpy = numpy.float32 # Default
    compiled_model = None

    try:
        print("Initializing OpenVINO Core...")
        core = ov.Core()
        print(f"Reading IR model from: {model_xml_path}")
        model = core.read_model(model=model_xml_path)

        if len(model.inputs) != 1:
            print(f"Warning: Model has {len(model.inputs)} inputs. This script primarily supports single input models.", file=sys.stderr)

        input_tensor_pre_reshape = model.input(0)
        desired_input_shape_list = [args.batch_size, 3, args.img_size, args.img_size]

        if args.static_shapes:
            try:
                model.reshape({input_tensor_pre_reshape.any_name: ov.PartialShape(desired_input_shape_list)})
                print(f"Reshape succeeded: {desired_input_shape_list}")
            except Exception as e:
                print(f"Reshape failed: {e}", file=sys.stderr)
        else:
            print(f"Keeping model input shape: {input_tensor_pre_reshape.partial_shape}")

        print(f"Compiling model for device: {args.device.upper()}...")
        compile_config = {'PERFORMANCE_HINT': 'LATENCY'}
        compiled_model = core.compile_model(model, device_name=args.device.upper(), config=compile_config)
        final_input_layer = compiled_model.input(0)
        final_input_shape_tuple = (
            tuple(desired_input_shape_list)
            if final_input_layer.partial_shape.is_dynamic
            else tuple(final_input_layer.shape)
        )
        element_type_numpy = final_input_layer.element_type.to_dtype()
        print(f"Model compiled. Final expected input shape: {final_input_shape_tuple}, Data type: {element_type_numpy}")

    except Exception as e:
        print(f"Error during OpenVINO model loading/compilation: {e}", file=sys.stderr)
        return

    # Data Loading
    all_processed_images_np = []
    if args.dataset_path:
        print(f"Loading dataset from: {args.dataset_path}")
        try:
            for img_processed in iter_benchmark_images(args.dataset_path, args.img_size):
                all_processed_images_np.append(img_processed.astype(element_type_numpy))

            if not all_processed_images_np:
                print(f"Error: No images found/loaded from: {args.dataset_path}", file=sys.stderr)
                return
            print(f"Loaded {len(all_processed_images_np)} images, preprocessed as CHW uint8 arrays.")
        except Exception as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
            return
    else:
        print(f"Using dummy data for benchmarking. Shape: {final_input_shape_tuple}, Type: {element_type_numpy}")
        if numpy.issubdtype(element_type_numpy, numpy.floating):
            dummy_input_batch_np = numpy.random.rand(*final_input_shape_tuple).astype(element_type_numpy)
        else:
            dummy_input_batch_np = numpy.random.randint(0, 256, size=final_input_shape_tuple, dtype=numpy.uint8).astype(element_type_numpy)

    # Batch provider for real data
    current_image_idx = 0
    def get_real_batch_from_preloaded_np():
        nonlocal current_image_idx
        if final_input_shape_tuple is None:
            raise ValueError("Model's final input shape not determined.")

        batch_frames_np = numpy.zeros(final_input_shape_tuple, dtype=element_type_numpy)
        for i in range(args.batch_size):
            img_np = all_processed_images_np[current_image_idx % len(all_processed_images_np)]
            if img_np.shape != final_input_shape_tuple[1:]:
                 print(f"Warning: Image shape {img_np.shape} does not match model's expected spatial/channel dims {final_input_shape_tuple[1:]}. Attempting resize.", file=sys.stderr)
                 img_resized = cv2.resize(img_np.transpose(1,2,0), (final_input_shape_tuple[3], final_input_shape_tuple[2])).transpose(2,0,1)
                 batch_frames_np[i] = img_resized.astype(element_type_numpy) # Ensure type after resize
            else:
                batch_frames_np[i] = img_np
            current_image_idx += 1
        return batch_frames_np

    # Bind application-created tensors so dynamic NPU requests receive the
    # concrete batch shape and rebuild their output pipeline correctly.
    infer_request = compiled_model.create_infer_request()

    def bind_input_tensor(batch_data_np):
        if tuple(batch_data_np.shape) != final_input_shape_tuple:
            raise ValueError(
                f"Input batch shape {tuple(batch_data_np.shape)} does not match "
                f"expected shape {final_input_shape_tuple}."
            )
        infer_request.set_input_tensor(ov.Tensor(batch_data_np))

    def verify_output_batch_size():
        output_batches = {
            infer_request.get_output_tensor(index).shape[0]
            for index in range(len(compiled_model.outputs))
        }
        if output_batches != {args.batch_size}:
            raise RuntimeError(
                f"Output batch sizes {sorted(output_batches)} do not match "
                f"input batch size {args.batch_size}."
            )

    output_batch_verified = False

    # Warm-up Phase
    try:
        print(f"\nStarting {args.warmup_runs} warm-up runs (OpenVINO IR model)...")
        for _ in range(args.warmup_runs):
            if args.dataset_path:
                batch_data_np = get_real_batch_from_preloaded_np()
            else:
                batch_data_np = dummy_input_batch_np
            bind_input_tensor(batch_data_np)
            infer_request.start_async()
            infer_request.wait()
        if args.warmup_runs:
            verify_output_batch_size()
            output_batch_verified = True
        print("Warm-up runs completed.")
    except Exception as e:
        print(f"Error during warm-up runs: {e}", file=sys.stderr)
        return

    # Benchmark Phase
    try:
        print(f"\nStarting {args.benchmark_runs} benchmark runs...")
        timings = []
        device_timings = []
        for i in range(args.benchmark_runs):
            if args.dataset_path:
                batch_data_np = get_real_batch_from_preloaded_np()
            else:
                batch_data_np = dummy_input_batch_np
            bind_input_tensor(batch_data_np)
            start_time = time.perf_counter()
            infer_request.start_async()
            infer_request.wait()
            end_time = time.perf_counter()
            if not output_batch_verified:
                verify_output_batch_size()
                output_batch_verified = True
            timings.append(end_time - start_time)
            device_timings.append(infer_request.latency)
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{args.benchmark_runs} benchmark runs...")
        print("Benchmark runs completed.")
    except Exception as e:
        print(f"Error during benchmark runs: {e}", file=sys.stderr)
        return

    # Results Calculation & Output
    if not timings:
        print("No benchmark timings recorded.", file=sys.stderr)
        return

    avg_time_s = sum(timings) / len(timings)
    avg_time_ms = avg_time_s * 1000
    median_time_ms = numpy.median(timings) * 1000
    median_device_latency_ms = numpy.median(device_timings)
    fps = 1 / avg_time_s if avg_time_s > 0 else 0

    print("\n--- OpenVINO IR Benchmark Results ---")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path if args.dataset_path else 'Dummy Data'}")
    print(f"Input Shape (used for benchmark, from compiled model): {final_input_shape_tuple}")
    print(f"Device: {args.device.upper()}")
    print(f"Number of warm-up runs: {args.warmup_runs}")
    print(f"Number of benchmark runs: {args.benchmark_runs}")
    print("--------------------------------------")
    print(f"Average wall-clock inference time: {avg_time_ms:.2f} ms")
    print(f"Median wall-clock inference time: {median_time_ms:.2f} ms")
    print(f"Median OpenVINO device latency: {median_device_latency_ms:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("--------------------------------------\n")

if __name__ == '__main__':
    main()
