# YOLO3D on an AI PC: One Pipeline, Three Device Choices with OpenVINO

An AI PC can have several places to run a neural network: a CPU, a GPU, and an NPU. For a developer, that raises a practical question: can you use those devices without maintaining a different version of your application for each one?

With OpenVINO, you can keep a common model format and inference API while choosing a supported execution device. In this project, we brought [YOLO3D](https://github.com/ruhyadi/YOLO3D) into that workflow, taking a camera image through object detection and into projected 3D bounding boxes.

Let's walk through what that means, how the pieces fit together, and how to start trying the devices on your own PC.

## From a Camera Image to a 3D Box

A 2D detector tells you where an object appears in an image. A 3D detection pipeline goes a step further: it estimates the object's dimensions and orientation, then uses camera geometry to construct a 3D box.

Despite its name, YOLO3D is not a single neural network. It combines two networks with geometric calculations:

1. **YOLOv5s finds objects.** It identifies objects in the image and returns 2D bounding boxes.
2. **A ResNet18-based regressor examines object crops.** It predicts orientation information and dimension offsets.
3. **Geometry brings the predictions together.** Camera calibration and class-average dimensions help turn the outputs into projected 3D boxes.

For example, YOLO finds a car in a street scene. The regressor estimates its size and heading. The geometric code uses those estimates and the camera calibration to draw a box around it in perspective.

Here is an existing OpenVINO example output:

![Projected YOLO3D bounding boxes in a KITTI driving scene](https://raw.githubusercontent.com/avidiyal/YOLO3D/main/runs_xml/000.png)

These are estimates from a single image, not direct depth measurements. The camera calibration is part of the pipeline, not an optional finishing touch.

## Meet the Three Execution Targets

The devices for OpenVINO inference:

| Device | What it is | Its place in this project |
|---|---|---|
| CPU | The computer's general-purpose processor. | A starting point for running and validating the OpenVINO models. |
| GPU | A processor commonly associated with graphics that can also execute neural networks. | Another target for the detector and regressor through OpenVINO's GPU plugin. |
| NPU | A neural processing unit designed for AI workloads. | A dedicated AI target for supported model and input configurations. |

Using a different device is as simple as setting a device parameter. No need to change anything else in the code.

## Prepare the Models Once

We start with PyTorch checkpoints for YOLOv5s and the ResNet18-based regressor. We use `ov.convert_model`, OpenVINO's Python conversion function, to turn each network into an OpenVINO model. It accepts a supported model object or model file; see the [ov.convert_model documentation](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.convert_model.html) for the available options. We then use `ov.save_model` to save the result in OpenVINO Intermediate Representation, usually called IR.

The repo's [conversion script](https://github.com/avidiyal/YOLO3D/blob/main/weights/convert_models.py) contains the code to export both models to OpenVINO. It converts the loaded PyTorch models using `ov.convert_model`.

After following the [OpenVINO setup guide](https://github.com/avidiyal/YOLO3D/blob/main/README_openvino.md) to install dependencies and download both source checkpoints, run this command from the repository root:

```powershell
python weights/convert_models.py
```

With the default paths, the script writes `yolov5s.xml` and `resnet18.xml` into `weights/openvino/`, each with a matching `.bin` file. The `.xml` describes the model graph, and the `.bin` contains its weights. Keep each pair together when moving the models to another machine.

Those converted models become a common starting point for device selection. You do not need to change anything to compile for another supported device. 

### Include Preprocessing in the Model

Before a network can use an image, it may need steps such as changing the color order or scaling pixel values. OpenVINO's [preprocessing API](https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html) lets you add these steps to the model itself using `PrePostProcessor`. This saves writing separate preprocessing code in the application, and the included steps run as part of the model graph on the selected target device.

In the convert_models.py script, we do this for the ResNet18-based regressor, including conversion from integer pixel values and normalization. 

## Ask OpenVINO What Is Available

After setup, this small example lists the devices OpenVINO can see:

```python
import openvino as ov

core = ov.Core()
for device in core.available_devices:
    print(device, core.get_property(device, "FULL_DEVICE_NAME"))
```

Depending on your machine, you might see identifiers such as `CPU`, `GPU.0`, and `NPU`. Use the identifiers returned on your PC rather than assuming all three will be present.

If an expected device is missing, check the [OpenVINO installation and driver requirements](https://docs.openvino.ai/2026/get-started/install-openvino/configurations.html). Device discovery tells you that the runtime can see a device; compiling your model is the next compatibility check.

## Choose a Device with the Same API

Run this example from the repository root after converting the models:

```python
import openvino as ov

core = ov.Core()
device = "CPU"

model = core.read_model("weights/openvino/resnet18.xml")
model.reshape({model.input(0): [1, 3, 224, 224]})
compiled = core.compile_model(model, device)

print("Execution device:", compiled.get_property("EXECUTION_DEVICES"))
```

The shape means one object crop, three color channels, and a height and width of 224 pixels. Keeping this example at a fixed batch of one makes the input contract explicit.

To try a different target, change the device assignment to an available identifier:

```python
device = "GPU.0"
```

Or, on a machine with a supported NPU:

```python
device = "NPU"
```

Then rerun the model-loading and compilation example with that device. The model path and compilation API stay the same; the selected execution target changes. 

## Keep the Inputs and Geometry Consistent

Device selection is only one part of a working application. The input must still match the model.

The repo's [XML inference demo](https://github.com/avidiyal/YOLO3D/blob/main/inference_scripts/inference_openvino_xml.py) shows how the networks connect to cropping, calibration, and box drawing. That demo currently hardcodes CPU execution, so it is a useful pipeline reference rather than a ready-made CPU/GPU/NPU command-line switch. The examples above show device selection directly through the OpenVINO API.

Keep the camera calibration with the images it describes. Changing an execution device does not change the camera's geometry or remove the need for the correct calibration file.

## Two Networks Also Mean Two Placement Choices

Because YOLO and the regressor are separate models, an application can compile them independently. Both could use CPU, both could use GPU, or an application could place the detector on GPU and the regressor on NPU.

## Flexibility Still Needs Validation

Successful compilation and inference are only the beginning: It is always important to check that the detections, orientation estimates, dimensions, and projected boxes remain meaningful for your images and camera calibration. The repo's [accuracy evaluator](https://github.com/avidiyal/YOLO3D/blob/main/inference_scripts/accuracy.py) compares source and OpenVINO models, while the [XML inference demo](https://github.com/avidiyal/YOLO3D/blob/main/inference_scripts/inference_openvino_xml.py) lets you inspect rendered 3D boxes on CPU.

An approachable validation routine is:

1. **Start with a known input on CPU.** Check the detections and projected boxes using the correct camera calibration.
2. **Compute accuracy on other devices.** 
3. **Check the outputs.** Verify output shapes, a result for every object crop, and sensible orientation and dimension estimates. Small numerical differences are possible; compare against defined tolerances rather than expecting identical values.
4. **Inspect the final boxes.** Valid tensor shapes alone cannot tell you whether the geometry is right.

The [accuracy workflow](https://github.com/avidiyal/YOLO3D/blob/main/README_openvino.md#7-measure-source-and-openvino-accuracy) describes comparisons between source and OpenVINO models. It treats 2D detection and 3D regression as separate tasks, which helps identify where a difference comes from.

## Exploring Quantization

Quantization is a way to make parts of a model use less detailed numbers, which may speed up inference. For example, INT8 represents values using 8-bit integers instead of more precise floating-point numbers. The repo's [ResNet quantization script](https://github.com/avidiyal/YOLO3D/blob/main/weights/quantize_resnet.py) uses Intel's [NNCF](
https://github.com/openvinotoolkit/nncf) to prepare this version of the model. You provide example images of cropped objects so the toolkit can choose suitable number ranges, a step called calibration. This is different from camera calibration, which describes the camera's geometry. Using less detailed numbers can change predictions, so check the quantized model with the accuracy evaluator and inspect its 3D boxes. Use a separate set of images for those checks where possible.

## One Pipeline, More Places to Run It

The appeal of OpenVINO in this YOLO3D project is the ability to keep the application recognizable as the hardware target changes. The same two-network structure, model artifacts, camera calibration, and geometric logic remain at the center of the workflow.

Start by asking which devices your PC can see. Load a converted model, choose a target, and check its outputs. Then connect that validated configuration to the rest of the 3D pipeline.

CPU, GPU, and NPU become choices you can explore through a shared interface, rather than three separate versions of the project.
