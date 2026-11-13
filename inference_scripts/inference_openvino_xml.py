# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""

import argparse
import os
import sys
from pathlib import Path
import glob

import cv2
import torch
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.datasets import LoadImages
from utils.general import LOGGER, check_requirements, non_max_suppression, print_args, scale_coords
from utils.torch_utils import time_sync

import torch
import torch.nn as nn

import numpy as np

from script.Dataset import generate_bins, DetectedObject
from library.Math import *
from library.Plotting import *
from script import Model, ClassAverages
import openvino as ov

core = ov.Core()


class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_

def detect3d(
    reg_weights,
    source,
    calib_file,
    show_result,
    save_result,
    output_path,
    yolo_weights='../weights/openvino/yolov5s.xml',
    yolo_data='../data/coco128.yaml',
    det_device='CPU',
    reg_device='CPU'
    ):

    # Directory
    imgs_path = sorted(glob.glob(str(source) + '/*'))
    calib = str(calib_file)

    # Compile each model once. Static input shapes let the same IR compile on CPU, GPU, and NPU.
    regressor_model = core.read_model(reg_weights)
    regressor_model.reshape({regressor_model.input(0): [1, 3, 224, 224]})
    compiled_regressor = core.compile_model(regressor_model, reg_device)

    yolo_model = core.read_model(yolo_weights)
    yolo_model.reshape({yolo_model.input(0): [1, 3, 640, 640]})
    compiled_yolo = core.compile_model(yolo_model, det_device)
    with open(yolo_data, errors='ignore') as f:
        yolo_names = yaml.safe_load(f)['names']

    LOGGER.info(f'YOLO detector on {compiled_yolo.get_property("EXECUTION_DEVICES")}, '
                f'regressor on {compiled_regressor.get_property("EXECUTION_DEVICES")}')

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    # loop images
    for i, img_path in enumerate(imgs_path):
        # read image
        img = cv2.imread(img_path)
        
        # Run detection 2d
        dets = detect2d(
            compiled_model=compiled_yolo,
            names=yolo_names,
            source=img_path,
            imgsz=[640, 640],
            classes=[0, 2, 3, 5]
        )

        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue
            try: 
                detectedObject = DetectedObject(img, det.detected_class, det.box_2d, calib)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img_uint8
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class

            # predict orient, conf, and dim
            results = compiled_regressor(input_img[None])
            orient = results[compiled_regressor.output(0)]
            conf = results[compiled_regressor.output(1)]
            dim = results[compiled_regressor.output(2)]

            # No need for .cpu().data.numpy() - already NumPy arrays
            orient = orient[0, :, :]  
            conf = conf[0, :]
            dim = dim[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            # plot 3d detection
            plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)

        if show_result:
            cv2.imshow('3d detection', img)
            cv2.waitKey(0)
        
        if save_result and output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            # Save the image
            im_name = f'{output_path}/{i:03d}.png'
            cv2.imwrite(im_name, img)
            LOGGER.info(f'Output Saved {im_name}')

@torch.no_grad()
def detect2d(
    compiled_model,
    names,
    source,
    imgsz,
    classes
    ):

    # array for boundingbox detection
    bbox_list = []

    # Directories
    source = str(source)

    # Dataloader: letterbox to the fixed model input size
    dataset = LoadImages(source, img_size=imgsz, stride=32, auto=False)

    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        results = compiled_model(im.numpy())  # Inference with OpenVINO
        # Inference
        # pred = model(im, augment=False, visualize=False)
        # pred = compiled_model(im.numpy())  # Inference with OpenVINO
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = results[compiled_model.output(0)]  # Access the first output
        pred = torch.tensor(pred)

        # NMS
        pred = non_max_suppression(prediction=pred, classes=classes)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = (torch.tensor(xyxy).view(1,4)).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    top_left, bottom_right = (xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])
                    bbox = [top_left, bottom_right]
                    c = int(cls)  # integer class
                    label = names[c]

                    bbox_list.append(Bbox(bbox, label))

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    return bbox_list

def plot3d(
    img,
    proj_matrix,
    box_2d,
    dimensions,
    alpha,
    theta_ray,
    img_2d=None
    ):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location) # 3d boxes

    return location

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=str(ROOT / 'weights/openvino/yolov5s.xml'), help='model path(s)')
    parser.add_argument('--source', type=str, default=str(ROOT / 'eval/image_2'), help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=str(ROOT / 'data/coco128.yaml'), help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='CPU', help='OpenVINO device for both models, e.g. CPU, GPU, NPU')
    parser.add_argument('--det_device', default=None, help='OpenVINO device for the YOLO detector (overrides --device)')
    parser.add_argument('--reg_device', default=None, help='OpenVINO device for the regressor (overrides --device)')
    parser.add_argument('--classes', default=[0, 2, 3, 5], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--reg_weights', type=str, default=str(ROOT / 'weights/openvino/resnet18.xml'), help='Regressor model weights')
    parser.add_argument('--calib_file', type=str, default=str(ROOT / 'eval/camera_cal/calib_cam_to_cam.txt'), help='Calibration file or path')
    parser.add_argument('--show_result', action='store_true', help='Show Results with imshow')
    parser.add_argument('--save_result', action='store_true', help='Save result')
    parser.add_argument('--output_path', type=str, default=str(ROOT / 'output'), help='Save output pat')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    detect3d(
        reg_weights=opt.reg_weights,
        source=opt.source,
        calib_file=opt.calib_file,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path,
        yolo_weights=opt.weights[0] if isinstance(opt.weights, list) else opt.weights,
        yolo_data=opt.data,
        det_device=opt.det_device or opt.device,
        reg_device=opt.reg_device or opt.device
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
