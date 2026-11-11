# YOLO3D with OpenVINO: Convert Once, Run on CPU, GPU, or NPU

Say you're adding a vision feature to a desktop application. By the time your code starts, a local LLM may already be using the GPU, and a video call may be keeping the NPU busy with background blur. So where does your model run? With OpenVINO™, you don't have to decide up front.

You convert the model once, and the device it runs on is just a string you pass when you load it. To show what that looks like in practice, we took [YOLO3D](https://github.com/ruhyadi/YOLO3D), a small 3D object detection pipeline built from two neural networks, and ran it on the CPU, GPU, and NPU of an Intel® Core™ Ultra AI PC without changing the code. You don't need that exact machine to follow along. The same steps work on any Intel CPU, GPU, or NPU that OpenVINO supports.

**What you'll see:**

- One set of OpenVINO model files runs on the CPU, the GPU, and the NPU.
- Switching devices is a one-word change, and the two models in the pipeline can each use a different device.
- On an Intel Core Ultra processor (Series 2) AI PC, the full pipeline ran in real time on both the GPU and the NPU. The CPU runs it too, and it's the fallback you always have.¹
- Quantizing the models to INT8 takes two commands.

## How the Pipeline Works

YOLO3D is two small networks with some ordinary code between them, which is how a lot of real vision applications are built:

1. **YOLOv5s finds the objects.** It returns a 2D bounding box for each one.
2. **A ResNet18-based regressor looks at each object crop.** It predicts which way the object is facing and how its size differs from the average for its class.
3. **Geometry does the rest.** The code combines those predictions with the camera calibration to get 3D boxes, then draws them on the image.

![Projected YOLO3D bounding boxes in a KITTI driving scene](https://raw.githubusercontent.com/avidiyal/YOLO3D/main/runs_xml/000.png)

*Output of the OpenVINO pipeline on a scene from the [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/). The boxes are estimates from a single image, not depth measurements.*

Having two models is what makes this a useful example, because later we put them on different devices. None of the steps below are specific to YOLO3D, so you can reuse them for your own models.

## Step by Step: Convert, Quantize, and Run

All you need to follow along is an Intel CPU, Git, and Python 3 (we used 3.12). If your system also has an Intel GPU or NPU, [install the drivers](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html) so OpenVINO can use them. An Intel Core Ultra AI PC has all three devices.

Run everything after step 1 from the `YOLO3D` folder. The commands are written for Linux. On Windows, use the alternatives in the comments, and end continued lines with `^` (Command Prompt) or a backtick (PowerShell) instead of `\`.

**1. Get the code and install the dependencies.**

```bash
git clone https://github.com/avidiyal/YOLO3D.git
cd YOLO3D
python3 -m venv ov-yolo3d-env          # Windows: python -m venv ov-yolo3d-env
source ov-yolo3d-env/bin/activate      # Windows: ov-yolo3d-env\Scripts\activate
pip install -r requirements-cpu.txt
```

**2. Download the PyTorch checkpoints and convert them to OpenVINO.**

You only need to do this once.

```bash
python weights/get_weights.py
python weights/convert_models.py
```

This writes `yolov5s.xml` and `resnet18.xml` to `weights/openvino/`, each with a matching `.bin` file.

**3. Optional: quantize the models to INT8.**

```bash
python weights/quantize_yolo.py \
  --ir weights/openvino/yolov5s.xml \
  --images eval/image_2/ \
  --output weights/openvino/yolov5s_int8.xml

python weights/quantize_resnet.py
```

That's all quantization takes: two commands and about a minute, with no retraining. [NNCF](https://github.com/openvinotoolkit/nncf) does the work. INT8 models boost performance, especially on the CPU and the NPU, with minimal accuracy loss. You'll find them next to the originals as `yolov5s_int8.xml` and `resnet18_int8.xml`, and they run on every device. Steps 5 and 7 show how to use them.

**4. Check which devices OpenVINO can see.**

```bash
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

Use the names exactly as they're printed. If the system has more than one GPU, they show up as `GPU.0`, `GPU.1`, and so on, and plain `GPU` means `GPU.0`.

**5. Run the 3D pipeline on each device.**

```bash
python inference_scripts/inference_openvino_xml.py --device CPU \
  --save_result --output_path output_cpu

python inference_scripts/inference_openvino_xml.py --device GPU \
  --save_result --output_path output_gpu

python inference_scripts/inference_openvino_xml.py --device NPU \
  --save_result --output_path output_npu
```

`--device` is the only thing that changes between these commands. Each run logs which device the models are on and saves the annotated images to its output folder. Compare the folders and you'll see the same objects detected on every device.

If you did step 3, add the INT8 model paths to run the quantized models. This works on any device:

```bash
python inference_scripts/inference_openvino_xml.py --device NPU \
  --weights weights/openvino/yolov5s_int8.xml \
  --reg_weights weights/openvino/resnet18_int8.xml \
  --save_result --output_path output_npu_int8
```

**6. Split the pipeline across two devices.**

```bash
python inference_scripts/inference_openvino_xml.py \
  --det_device GPU --reg_device NPU \
  --save_result --output_path output_split
```

This puts the detector on the GPU and the 3D regressor on the NPU. Any two devices from step 4 will work.

**7. Measure the speed on your hardware.**

```bash
python inference_scripts/benchmark_full_pipeline.py \
  --pipeline_type ir_openvino \
  --yolo_weights weights/openvino/yolov5s.xml \
  --regressor_weights weights/openvino/resnet18.xml \
  --dataset_path eval/image_2/ \
  --calib_file eval/camera_cal/calib_cam_to_cam.txt \
  --device NPU
```

The script prints the time spent in each stage and the overall frames per second. Change `--device` to compare devices. To measure the INT8 models from step 3, pass `yolov5s_int8.xml` and `resnet18_int8.xml` instead.

## Balance the Work Across Devices

Because the same model files compile for any device, you can put each part of the pipeline wherever there's spare capacity, and move it later without touching the rest of your application. For example:

| What's happening on the system | Where to run the 3D pipeline |
|---|---|
| The GPU is busy with a local LLM, a game, or other graphics work | The NPU, or the NPU and CPU together |
| The NPU is busy with video-call effects or other background AI | The GPU |
| The system is on battery for a long session | The NPU, which is designed for sustained, low-power AI |
| The CPU is busy with a software build or a video export | The GPU or the NPU |

The detector and the regressor are separate models, so they don't have to share a device. This is what the split from step 6 looks like in code. Run it from the `YOLO3D` folder:

```python
import openvino as ov

core = ov.Core()
core.set_property({"CACHE_DIR": "model_cache"})  # reuse compiled models on later runs

def load(path, shape, device):
    model = core.read_model(path)
    # static shapes work on every device, including the NPU
    model.reshape({model.input(0): shape})
    return core.compile_model(model, device)

detector = load("weights/openvino/yolov5s.xml", [1, 3, 640, 640], "GPU")
regressor = load("weights/openvino/resnet18.xml", [1, 3, 224, 224], "NPU")
print(detector.get_property("EXECUTION_DEVICES"))
print(regressor.get_property("EXECUTION_DEVICES"))
```

To move the regressor to the CPU, change `"NPU"` to `"CPU"`. Two more OpenVINO features help once you start moving models around:

- **Model caching.** With `CACHE_DIR` set, OpenVINO saves each compiled model and reuses it the next time, so your application starts faster. See [model caching](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html).
- **Automatic device selection.** If you pass `"AUTO:NPU,GPU,CPU"` as the device name, OpenVINO picks the first device in the list that's available. See [automatic device selection](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html).

## Convert Once, Validate Everywhere

The [conversion script](https://github.com/avidiyal/YOLO3D/blob/main/weights/convert_models.py) calls [`ov.convert_model`](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.convert_model.html) on each PyTorch model directly, so there's no ONNX export in the middle. The `.xml` and `.bin` files it writes are the only model files you need, whatever device you run on. For the regressor, the script also uses the OpenVINO [preprocessing API](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html) to move input conversion and normalization into the model, which means that work runs on your chosen device as well.

Don't expect bit-identical results across devices. By default, the GPU and NPU typically compute in 16-bit floating point, and the CPU typically uses 32-bit. In our runs, every device found the same objects, and the 3D boxes differed only slightly. To put numbers on it, the [accuracy evaluator](https://github.com/avidiyal/YOLO3D/blob/main/inference_scripts/accuracy.py) compares the original PyTorch model with the OpenVINO model on labeled KITTI images:

```bash
python inference_scripts/accuracy.py resnet \
  --source-model weights/resnet18.pkl \
  --openvino-model weights/openvino/resnet18.xml \
  --hf-dataset nateraw/kitti \
  --limit 100 \
  --device NPU
```

Run the same check on the INT8 models before you rely on them. The [accuracy workflow](https://github.com/avidiyal/YOLO3D/blob/main/README_openvino.md#7-measure-source-and-openvino-accuracy) has the details.

## Wrapping Up

It took one conversion step and one API to run YOLO3D on three devices, and moving a model between them meant changing a string. That leaves you free to decide where vision runs based on what else the system is doing.

- [Follow the steps above](#step-by-step-convert-quantize-and-run) and measure the pipeline on your own hardware.
- Try the same approach with your own models. The [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks) have plenty of examples to start from.
- Browse the code, conversion scripts, and benchmark scripts in the [YOLO3D repository](https://github.com/avidiyal/YOLO3D).

---

## Notices and Credits

¹ Test configuration: Intel® Core™ Ultra 5 processor 236V with integrated Intel® Arc™ GPU and Intel® AI Boost NPU, 16 GB memory, Ubuntu 24.04, OpenVINO 2026.4.1, Intel NPU driver 1.38, Intel GPU compute runtime 26.40. The full pipeline was measured with `benchmark_full_pipeline.py` on the sample KITTI images in the repository, with the FP32 models. Real time means faster than the frame rate of the camera that recorded the KITTI driving data. Tested in November 2026. Performance varies by use, configuration and other factors. Learn more at [www.intel.com/PerformanceIndex](https://www.intel.com/PerformanceIndex).

- YOLO3D implementation by [ruhyadi](https://github.com/ruhyadi/YOLO3D), based on *3D Bounding Box Estimation Using Deep Learning and Geometry* by [Mousavian et al.](https://arxiv.org/abs/1612.00496)
- YOLOv5 by [Ultralytics](https://github.com/ultralytics/yolov5), licensed under GPL-3.0.
- Example image from the [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/) by Geiger et al., licensed under CC BY-NC-SA 3.0.
- Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.
