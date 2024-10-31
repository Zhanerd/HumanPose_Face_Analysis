import os
from abc import ABCMeta, abstractmethod
from typing import Any
import torch
import numpy as np


class BaseTool(metaclass=ABCMeta):
    def __init__(self,
                 backend: str = "onnxruntime",  #### just support onnx and rt now
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

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def release(self):
        del self.session

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # build input to (B, C, H, W)
        # img = img.transpose(2, 0, 1)
        # img = np.ascontiguousarray(img, dtype=np.float32)
        # input = img[None, :, :, :]

        img = img.transpose(0,3, 1, 2)
        input = np.ascontiguousarray(img, dtype=np.float32)

        # run model
        if self.backend == "onnxruntime":
            sess_input = {self.session.get_inputs()[0].name: input}
            sess_output = []
            for out in self.session.get_outputs():
                sess_output.append(out.name)
            outputs = self.session.run(sess_output, sess_input)
        elif self.backend == "tensorrt":
            input = torch.from_numpy(input).to('cuda')
            with torch.no_grad():
                outputs = self.session(input)
            outputs = [output.cpu().numpy() for output in outputs]
        return outputs
