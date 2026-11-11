# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Convert YOLO and ResNet checkpoints directly to OpenVINO IR.

Run from the repository root:
    python weights/convert_models.py
    python weights/convert_models.py --onnx

The --onnx flag additionally exports dynamic-batch ONNX models to weights/onnx/
(or --onnx_dir). IR models are still saved to weights/openvino/ (or --ir_dir)
using direct PyTorch conversion, not the ONNX exports. YOLO ONNX accepts RGB
float32 NCHW inputs scaled to [0, 1]; ResNet ONNX accepts ImageNet-normalized
float32 BGR NCHW crops. Only the ResNet IR includes uint8 input preprocessing.
"""

import argparse
import os
import sys
from pathlib import Path

# --- Add project root to sys.path ---
FILE_ROOT_SCRIPT = Path(__file__).resolve()
PROJECT_ROOT = FILE_ROOT_SCRIPT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to sys.path")

# --- Imports ---
import torch
import numpy as np
import openvino as ov

def load_pytorch_model(weights_path, cfg_path):
    """Load PyTorch YOLO model."""
    from models.yolo import Model as YOLOModel
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model = YOLOModel(cfg=cfg_path, ch=3, nc=None).cpu()
    
    model.load_state_dict(checkpoint['model'].state_dict())
    
    return model.eval()

def load_pkl_model(pkl_path):
    """Load PKL ResNet18 model."""
    from script.Model import ResNet18
    from torchvision.models import resnet18 as torchvision_resnet18
    
    base_model = torchvision_resnet18(weights=None, progress=False)
    model = ResNet18(model=base_model)
    checkpoint = torch.load(pkl_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.eval().cpu()

def convert_to_onnx(model, output_path, input_shape, input_names, output_names, dynamic_axes, opset_version):
    """Convert PyTorch model to ONNX."""
    dummy_input = torch.randn(*input_shape, device='cpu')
    
    torch.onnx.export(
        model, (dummy_input,), str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print(f"ONNX saved: {output_path}")

def add_resnet_preprocessing(model):
    """Expose uint8 NCHW crops while preserving ResNet's trained input values."""
    preprocessing = ov.preprocess.PrePostProcessor(model)
    preprocessing.input(0).model().set_layout(ov.Layout('NCHW'))
    preprocessing.input(0).tensor().set_layout(ov.Layout('NCHW'))
    preprocessing.input(0).tensor().set_element_type(ov.Type.u8)
    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
    scale = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
    preprocessing.input(0).preprocess().convert_element_type(ov.Type.f32).mean(mean).scale(scale)
    return preprocessing.build()


def convert_to_ir(onnx_path, ir_path, add_resnet_input_preprocessing=False, input_info=None):
    """Convert ONNX to OpenVINO IR."""
    conversion_args = {"input": input_info} if input_info is not None else {}
    ov_model = ov.convert_model(str(onnx_path), **conversion_args)
    if add_resnet_input_preprocessing:
        ov_model = add_resnet_preprocessing(ov_model)
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    print(f"IR saved: {ir_path}")

def convert_pytorch_to_ir(weights_path, cfg_path, onnx_dir, ir_dir, img_size, batch_size, opset_version, export_onnx=False):
    """Convert YOLO directly to IR, optionally saving a separate ONNX export."""
    from models.yolo import Detect

    print(f"\n--- Converting PyTorch Model ---")
    print(f"Weights: {weights_path}")
    
    model = load_pytorch_model(weights_path, cfg_path)
    detection_head = model.model[-1]
    if isinstance(detection_head, Detect):
        detection_head.onnx_dynamic = True
        detection_head.inplace = False
    
    model_name = Path(weights_path).stem
    ir_path = Path(ir_dir) / f"{model_name}.xml"
    
    input_shape = (batch_size, 3, *img_size)
    if export_onnx:
        if not isinstance(detection_head, Detect):
            raise TypeError("YOLO ONNX export requires a YOLOv5 Detect head.")
        output_names = ['output'] + [f'raw_{index}' for index in range(detection_head.nl)]
        convert_to_onnx(
            model, Path(onnx_dir) / f"{model_name}.onnx", input_shape,
            input_names=['images'],
            output_names=output_names,
            dynamic_axes={name: {0: 'batch_size'} for name in ['images', *output_names]},
            opset_version=opset_version,
        )
    dummy_input = torch.randn(*input_shape, device='cpu')
    ov_model = ov.convert_model(
        model,
        example_input=dummy_input,
        input=[ov.PartialShape([-1, 3, *img_size])],
    )
    ov_model.input(0).get_tensor().set_names({'images'})
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    print(f"IR saved: {ir_path}")

def convert_pkl_to_ir(pkl_path, onnx_dir, ir_dir, img_size, batch_size, opset_version, export_onnx=False):
    """Convert ResNet directly to IR, optionally exporting normalized-input ONNX."""
    print(f"\n--- Converting Regressor Model ---")
    print(f"PKL: {pkl_path}")
    
    model = load_pkl_model(pkl_path)
    
    model_name = Path(pkl_path).stem
    ir_path = Path(ir_dir) / f"{model_name}.xml"
    
    input_shape = (batch_size, 3, *img_size)
    if export_onnx:
        convert_to_onnx(
            model, Path(onnx_dir) / f"{model_name}.onnx", input_shape,
            input_names=['input'],
            output_names=['orient', 'conf', 'dim'],
            dynamic_axes={name: {0: 'batch_size'} for name in ['input', 'orient', 'conf', 'dim']},
            opset_version=opset_version,
        )
    dummy_input = torch.randn(*input_shape, device='cpu')
    ov_model = ov.convert_model(
        model,
        example_input=dummy_input,
        input=[ov.PartialShape([-1, 3, *img_size])],
    )
    ov_model.input(0).get_tensor().set_names({'input'})
    for output, name in zip(ov_model.outputs, ['orient', 'conf', 'dim']):
        output.get_tensor().set_names({name})
    ov_model = add_resnet_preprocessing(ov_model)
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    print(f"IR saved: {ir_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch (.pt) or PKL (.pkl) models directly to OpenVINO IR, optionally also exporting ONNX."
    )
    
    # Model paths
    parser.add_argument('--pytorch_weights', type=str, default=str(PROJECT_ROOT / "weights/yolov5s.pt"), help="PyTorch .pt weights file")
    parser.add_argument('--pytorch_cfg', type=str, default=str(PROJECT_ROOT / "models/yolov5s.yaml"), help="PyTorch .yaml config file")
    parser.add_argument('--pkl_path', type=str, default=str(PROJECT_ROOT / "weights/resnet18.pkl"), help="PKL model file")
    
    # Output directories
    parser.add_argument('--onnx', action='store_true', help='Also export dynamic-batch ONNX models to --onnx_dir; direct IR conversion remains enabled.')
    parser.add_argument('--onnx_dir', type=str, default=str(PROJECT_ROOT / "weights/onnx"))
    parser.add_argument('--ir_dir', type=str, default=str(PROJECT_ROOT / "weights/openvino"))
    parser.add_argument(
        '--convert_resnet_no_preprocessing_only',
        action='store_true',
        help='Convert the existing ResNet ONNX model to a separate IR without embedded preprocessing.',
    )
    parser.add_argument(
        '--resnet_no_preprocessing_ir',
        type=Path,
        default=PROJECT_ROOT / "weights/openvino/resnet18_no_preprocessing.xml",
        help='Output XML path for --convert_resnet_no_preprocessing_only. Refuses to overwrite an existing model.',
    )
    parser.add_argument(
        '--convert_resnet_no_preprocessing_bounded_only',
        action='store_true',
        help='Convert ResNet ONNX with conversion-time bounded batch input and without embedded preprocessing.',
    )
    parser.add_argument(
        '--resnet_no_preprocessing_bounded_ir',
        type=Path,
        default=PROJECT_ROOT / "weights/openvino/resnet18_no_preprocessing_conversion_bounded.xml",
        help='Output XML path for --convert_resnet_no_preprocessing_bounded_only. Refuses to overwrite an existing model.',
    )
    parser.add_argument(
        '--resnet_batch_max',
        type=int,
        default=16,
        help='Maximum batch size for --convert_resnet_no_preprocessing_bounded_only.',
    )
    
    # PyTorch parameters
    parser.add_argument('--img_size_pytorch', nargs=2, type=int, default=[640, 640])
    parser.add_argument('--batch_size_pytorch', type=int, default=1)
    parser.add_argument('--opset_version_pytorch', type=int, default=12)
    
    # PKL parameters
    parser.add_argument('--img_size_pkl', nargs=2, type=int, default=[224, 224])
    parser.add_argument('--batch_size_pkl', type=int, default=1)
    parser.add_argument('--opset_version_pkl', type=int, default=11)
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.onnx_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ir_dir).mkdir(parents=True, exist_ok=True)

    if args.convert_resnet_no_preprocessing_only and args.convert_resnet_no_preprocessing_bounded_only:
        parser.error(
            "--convert_resnet_no_preprocessing_only and "
            "--convert_resnet_no_preprocessing_bounded_only cannot be used together"
        )

    if args.onnx and (args.convert_resnet_no_preprocessing_only or args.convert_resnet_no_preprocessing_bounded_only):
        parser.error("--onnx cannot be combined with the ONNX-to-IR-only conversion modes")

    onnx_path = Path(args.onnx_dir) / f"{Path(args.pkl_path).stem}.onnx"
    if args.convert_resnet_no_preprocessing_only:
        if not onnx_path.is_file():
            parser.error(f"ResNet ONNX model does not exist: {onnx_path}")
        if args.resnet_no_preprocessing_ir.exists():
            parser.error(
                f"Refusing to overwrite existing IR: {args.resnet_no_preprocessing_ir}. "
                "Choose a different --resnet_no_preprocessing_ir path."
            )
        args.resnet_no_preprocessing_ir.parent.mkdir(parents=True, exist_ok=True)
        convert_to_ir(onnx_path, args.resnet_no_preprocessing_ir)
        return

    if args.convert_resnet_no_preprocessing_bounded_only:
        if not onnx_path.is_file():
            parser.error(f"ResNet ONNX model does not exist: {onnx_path}")
        if args.resnet_batch_max < 1:
            parser.error("--resnet_batch_max must be a positive integer")
        if args.resnet_no_preprocessing_bounded_ir.exists():
            parser.error(
                f"Refusing to overwrite existing IR: {args.resnet_no_preprocessing_bounded_ir}. "
                "Choose a different --resnet_no_preprocessing_bounded_ir path."
            )
        args.resnet_no_preprocessing_bounded_ir.parent.mkdir(parents=True, exist_ok=True)
        input_shape = ov.PartialShape([ov.Dimension(1, args.resnet_batch_max), 3, 224, 224])
        convert_to_ir(
            onnx_path,
            args.resnet_no_preprocessing_bounded_ir,
            input_info=[("input", input_shape)],
        )
        return

    # Convert models if files exist
    if Path(args.pytorch_weights).exists() and Path(args.pytorch_cfg).exists():
        convert_pytorch_to_ir(
            args.pytorch_weights, args.pytorch_cfg,
            args.onnx_dir, args.ir_dir,
            args.img_size_pytorch, args.batch_size_pytorch, args.opset_version_pytorch,
            export_onnx=args.onnx,
        )
    else:
        print(f"Skipping YOLO conversion: {args.pytorch_weights} or {args.pytorch_cfg} not found. "
              "Run weights/get_weights.py first.", file=sys.stderr)

    if Path(args.pkl_path).exists():
        convert_pkl_to_ir(
            args.pkl_path,
            args.onnx_dir, args.ir_dir,
            args.img_size_pkl, args.batch_size_pkl, args.opset_version_pkl,
            export_onnx=args.onnx,
        )
    else:
        print(f"Skipping regressor conversion: {args.pkl_path} not found. "
              "Run weights/get_weights.py first.", file=sys.stderr)

    print("\nConversion complete!")

if __name__ == '__main__':
    main()