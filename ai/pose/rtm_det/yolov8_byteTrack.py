from typing import List, Tuple
import torch
import torchvision
import cv2
import numpy as np
from ..nets import nn
from ..base import BaseTool
from .post_processing import multiclass_nmsv2
from .utils import util

class YOLOv8_byteTrack():
    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (640, 640),
                 nms_thr=0.52,
                 score_thr=0.7,
                 gpu_id: int = 0,
                 backend: str = "tensorrt"):
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.final_cls = list()
        self.backend = backend
        if backend == "onnxruntime":
            import onnxruntime as ort
            if self.gpu_id < 0:
                self.session = ort.InferenceSession(path_or_bytes=model_path,
                                                    providers=['CPUExecutionProvider'])
            else:
                self.session = ort.InferenceSession(path_or_bytes=model_path,
                                                    providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        elif backend == "tensorrt":
            import tensorrt as trt
            from ai.torch2trt import TRTModule
            logger = trt.Logger(trt.Logger.INFO)
            with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())

            input_names = []
            output_names = []
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i):
                    input_names.append(engine.get_binding_name(i))
                else:
                    output_names.append(engine.get_binding_name(i))
            self.session = TRTModule(engine,input_names=input_names, output_names=output_names)
        self.bytetrack = nn.BYTETracker(25)
    ##### yolov8的调用看需求增加cls参数，默认只检测人，物体id要查coco_cat
    def __call__(self, image: np.ndarray,score_thr:float):
        self.score_thr = score_thr
        size = 640
        boxes = []
        confidences = []
        object_classes = []

        shape = image.shape[:2]

        r = size / max(shape[0], shape[1])
        if r != 1:
            h, w = shape
            image = cv2.resize(image,
                                dsize=(int(w * r), int(h * r)),
                                interpolation=cv2.INTER_LINEAR)

        h, w = image.shape[:2]
        image, ratio, pad = util.resize(image, size)
        shapes = shape, ((h / shape[0], w / shape[1]), pad)
        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)
        sample = np.expand_dims(sample,axis=0)

        sample = sample.astype('float32')
        sample = sample / 255  # 0 - 255 to 0.0 - 1.0
        # # Inference
        # with torch.no_grad():
        #     outputs = model(sample)

        if self.backend == "onnxruntime":
            sess_input = {self.session.get_inputs()[0].name: sample}
            sess_output = []
            for out in self.session.get_outputs():
                sess_output.append(out.name)
            outputs = self.session.run(sess_output, sess_input)[0]
            outputs = torch.from_numpy(outputs)

        elif self.backend == "tensorrt":
            input = torch.from_numpy(sample).to('cuda')
            with torch.no_grad():
                outputs = self.session(input)
            # outputs = [output.cpu().numpy() for output in outputs]
        # NMS
        outputs = util.non_max_suppression(outputs, 0.001, 0.7)
        for i, output in enumerate(outputs):
            detections = output.clone()
            util.scale(detections[:, :4], sample[i].shape[1:], shapes[0], shapes[1])
            detections = detections.cpu().numpy()
            for detection in detections:
                x1, y1, x2, y2 = list(map(int, detection[:4]))
                boxes.append([x1, y1, x2, y2])
                confidences.append(detection[4])
                object_classes.append(detection[5])
        shape = np.array(confidences).shape
        object_classes =  np.zeros(shape)  
        outputs = self.bytetrack.update(np.array(boxes),
                                    np.array(confidences),
                                    object_classes)
        boxes = outputs[:,:4]
        confidences = outputs[:,5]
        cls = outputs[:,6]
        track_id = outputs[:,7]
        print('v8_n track_id',track_id)
        return boxes,confidences,cls,track_id

def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
