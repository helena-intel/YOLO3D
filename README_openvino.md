## Setup and Inference Steps

### 1. Create a virtual environment
   ```bash
   python -m venv ov-yolo3d-env
   source ov-yolo3d-env/bin/activate   # Windows: ov-yolo3d-env\Scripts\activate
   ```

### 2. Install dependencies
   ```bash
   pip install -r requirements-cpu.txt
   ```
   - For Intel GPU and NPU devices, make sure to [install appropriate device drivers](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html)

### 3. Download model weights
   This step downloads pre-trained model weights using the `get_weights.py` script. It saves `resnet18.pkl` and `yolov5s.pt` into the `weights/` directory, whichever directory you run it from.

   - **Default behavior (downloads ResNet18 PKL and YOLOv5s PT):**
     ```bash
     python weights/get_weights.py
     ```

### 4. Convert Source Models to OpenVINO IR

  This step uses `weights/convert_models.py` to convert the PyTorch YOLO `.pt` and ResNet `.pkl` checkpoints directly into OpenVINO IR (`.xml` and `.bin` files). Converted models are saved to `weights/openvino/` by default.

   - **Default Conversion (YOLOv5s and ResNet18):**
     To convert the default `yolov5s.pt` (using `models/yolov5s.yaml`) and `resnet18.pkl` run:
     ```bash
     python weights/convert_models.py
     ```

### 5. Quantize models to INT8 (optional)

   This step uses NNCF (Neural Network Compression Framework) to post-training quantize the OpenVINO IR models produced in step 4 down to INT8, which can speed up inference at a small accuracy cost. NNCF is installed with `requirements-cpu.txt`. Run the following commands from the repository root.

   - **Quantize YOLOv5s** (`weights/quantize_yolo.py`), using real calibration images from `eval/image_2/`:
     ```bash
     python weights/quantize_yolo.py \
       --ir weights/openvino/yolov5s.xml \
       --images eval/image_2/ \
       --output weights/openvino/yolov5s_int8.xml
     ```
     This writes `yolov5s_int8.xml` / `yolov5s_int8.bin` to `weights/openvino/`.

   - **Quantize ResNet18 regressor** (`weights/quantize_resnet.py`) with real labeled KITTI object crops streamed from Hugging Face. It reads `weights/openvino/resnet18.xml` and writes `weights/openvino/resnet18_int8.xml` by default:
     ```bash
     python weights/quantize_resnet.py --max-samples 300
     ```
     The crop preprocessing matches ResNet evaluation. By default calibration uses the final contiguous shard of approximately 1,000 training images, while `accuracy.py` evaluates the first 1,000 streamed images, keeping the two samples separate. Use `--tail-images`, `--hf-dataset`, `--hf-split`, `--ir`, and `--output` to change the partition, data source, or model paths.

   For details on benchmarking these quantized models, see the [Benchmarking Guide](README_benchmarking.md#benchmarking-int8-quantized-models).

### 6. Run inference scripts

   - **Original inference script**
     (This script might need updates if it doesn't use the new model paths, or it might be deprecated if focus is on OpenVINO scripts.)
     ```bash
     python inference.py
     ```

   - **OpenVINO Accelerated Scripts**
     ```bash
     cd inference_scripts
     ```
     - To run the converted YOLO and ResNet IR models using OpenVINO APIs:
       ```bash
       # Assumes yolov5s.xml and resnet18.xml were created in step 4.
       python inference_openvino_api.py --save_result
       ```
     - To run a model with OpenVINO IR format (XML/BIN):
       Example using `yolov5s.xml` and `resnet18.xml`:
       ```bash
       # Assumes yolov5s.xml and resnet18.xml were created in step 4.
       python inference_openvino_xml.py  --save_result
       ```
       Choose the OpenVINO device with `--device` (`CPU`, `GPU`, or `NPU`; default `CPU`). To place the two models on different devices, use `--det_device` for YOLO and `--reg_device` for the regressor:
       ```bash
       python inference_openvino_xml.py --device NPU --save_result --output_path ../output_npu
       python inference_openvino_xml.py --det_device GPU --reg_device NPU --save_result --output_path ../output_split
       ```
       To verify the actual INT8 quantized models produce correct 3D detections (not just faster timings), point `--weights`/`--reg_weights` at the quantized files from step 5:
       ```bash
       python inference_openvino_xml.py \
         --weights ../weights/openvino/yolov5s_int8.xml \
         --reg_weights ../weights/openvino/resnet18_int8.xml \
         --save_result --output_path ../output_int8
       ```

     - To leverage `torch.compile` for an easier path to OpenVINO acceleration:
       Example using `yolov5s.pt`:
       ```bash
       # Assumes weights/yolov5s.pt is available
       python inference_openvino_compile.py --save_result
       ```

### 7. Measure source and OpenVINO accuracy

  `accuracy.py` evaluates either model type alone or prints comparable results for both models. It is tested with local KITTI data and the Hugging Face `nateraw/kitti` dataset. Other datasets need an adapter that maps their images, boxes, class names, and (for ResNet) orientation/dimensions into the KITTI fields this evaluator expects. Changing only the dataset ID or column names is not enough unless the annotation schema is compatible.

  By default, it evaluates at most 1,000 images. Use a local labeled KITTI split with this layout:

   ```text
   <dataset>/image_2/000001.png
   <dataset>/label_2/000001.txt
   ```

  YOLO reports 2D detection precision, recall, mAP@0.5, and mAP@0.5:0.95. The supplied COCO model is evaluated only on matching KITTI `Car` and `Pedestrian` classes. ResNet uses ground-truth 2D boxes to crop objects, isolating 3D regression quality from YOLO. It reports orientation and height/width/length errors; classes without learned average dimensions are skipped. These are task-specific comparison metrics, not one universal accuracy score.

  Run the following commands from the repository root (`cd ..` if continuing from step 6).

   ```bash
   # Compare source YOLOv5 with its OpenVINO IR on CPU.
   python inference_scripts/accuracy.py yolo \
     --source-model weights/yolov5s.pt \
     --openvino-model weights/openvino/yolov5s.xml \
     --dataset /path/to/KITTI/training \
     --device CPU

   # Evaluate the source ResNet and OpenVINO ResNet on an NPU.
  python inference_scripts/accuracy.py resnet \
     --source-model weights/resnet18.pkl \
     --openvino-model weights/openvino/resnet18.xml \
     --dataset /path/to/KITTI/training \
     --device NPU
   ```

   To stream a labeled 1,000-image sample without downloading the entire corpus, use the Hugging Face KITTI training split:

   ```bash
  python inference_scripts/accuracy.py yolo \
     --source-model weights/yolov5s.pt \
     --openvino-model weights/openvino/yolov5s.xml \
     --hf-dataset nateraw/kitti \
     --device CPU
   ```

  This uses labeled training images because the official KITTI test annotations are not public. Pass `--limit 0` to evaluate every streamed image, or `--limit 100` for a faster smoke test. Hugging Face data streams in source order by default to keep memory usage bounded; `--shuffle-seed 42` opts into reproducible shuffled sampling when sufficient memory is available. Pass only `--source-model` or only `--openvino-model` to evaluate one model. `--device` applies only to the OpenVINO model; PyTorch runs on CPU so the comparison focuses on model outputs rather than execution devices.

## Benchmarking Performance

For detailed instructions on how to run benchmarks, please refer to the [Benchmarking Guide](README_benchmarking.md).
