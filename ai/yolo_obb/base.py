import os
from abc import ABCMeta, abstractmethod
from typing import Any
import torch
import numpy as np


class BaseTool(metaclass=ABCMeta):
    def __init__(self, #### just support onnx and rt now
                 model_path: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 gpu_id: int = 0):

        if not os.path.exists(model_path):
            print('model path dont exist')

        self.model_path = model_path
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.gpu_id = gpu_id
        self.backend = None
        self.onnx_input = None
        self.onnx_output = None
        if "onnx" in os.path.basename(model_path):
            import onnxruntime as ort
            self.backend = 'onnxruntime'
            if self.gpu_id < 0:
                self.session = ort.InferenceSession(path_or_bytes=model_path,
                                                    providers=['CPUExecutionProvider'])
            else:
                self.session = ort.InferenceSession(path_or_bytes=model_path,
                                                    providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        elif "engine" in os.path.basename(model_path):
            self.backend = 'tensorrt'
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

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def release(self):
        if self.backend == 'onnxruntime':
            del self.session
        elif self.backend == 'tensorrt':
            del self.session.engine
            del self.session.context
            import gc
            gc.collect()
        else:
            pass
        self.session = None

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # if img.ndim == 3:
        #     # build input to (B, C, H, W)
        #     img = img.transpose(2, 0, 1)
        #     img = np.ascontiguousarray(img, dtype=np.float32)
        #     input = img[None, :, :, :]
        # elif img.ndim == 4:
        #     img = img.transpose(0,3, 1, 2)
        #     input = np.ascontiguousarray(img, dtype=np.float32)
        # else:
        #     print('img type not support')
        #     return None
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]
        # run model
        if self.backend == "onnxruntime":
            onnx_input = {self.session.get_inputs()[0].name: input}
            onnx_output = []
            for out in self.session.get_outputs():
                onnx_output.append(out.name)
            outputs = self.session.run(onnx_output, onnx_input)
        elif self.backend == "tensorrt":
            input = torch.from_numpy(input).to('cuda')
            with torch.no_grad():
                outputs = self.session(input)
            outputs = [output.cpu().numpy() for output in outputs]
        else:
            print('backend not support')
            outputs = None
        return outputs
