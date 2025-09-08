import os.path

import numpy as np
import onnxruntime as rt
import cv2
import tensorrt as trt
from ai.torch2trt.torch2trt import TRTModule
import torch

def preprocess(img):
    img = cv2.resize(img, (64, 128))
    img = np.float32(img)
    img = img / 255.0
    # img = img.transpose(2, 1, 0)
    # img = np.expand_dims(img, axis=0)
    return img


class Extractor:
    def __init__(self, model_path,gpu_id) -> None:
        self.input_names = ["input_1"]
        self.output_names = ["output_1"]
        self.backend = None
        input_mean = 127.5
        input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_size = (64, 128)
        if "onnx" in os.path.basename(model_path):
            self.backend = "onnxruntime"
            if gpu_id >= 0:
                providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self.model = rt.InferenceSession(model_path, providers=providers)
        elif "engine" in os.path.basename(model_path):
            self.backend = "tensorrt"
            logger = trt.Logger(trt.Logger.INFO)
            with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            input_names = []
            output_names = []
            if trt.__version__.split('.')[0] == '10':
                # rt10.x
                for name in engine:
                    mode = engine.get_tensor_mode(name)
                    if mode == trt.TensorIOMode.INPUT:
                        input_names.append(name)
                    elif mode == trt.TensorIOMode.OUTPUT:
                        output_names.append(name)
            else:
                # rt8.x
                for i in range(engine.num_bindings):
                    if engine.binding_is_input(i):
                        input_names.append(engine.get_binding_name(i))
                    else:
                        output_names.append(engine.get_binding_name(i))
            self.model = TRTModule(engine,input_names=input_names, output_names=output_names)

    def __call__(self, im_crops):
        imgs = []
        for im in im_crops:
            if im is None:
                print("im is None")
                continue
            inp = preprocess(im)
            imgs.append(inp)

        blob = cv2.dnn.blobFromImages(imgs, scalefactor=1.0 / self.input_std, size=self.input_size,mean=[self.input_mean, self.input_mean, self.input_mean], swapRB=True)
        blob = blob.transpose(0, 1, 3, 2)
        if self.backend == 'tensorrt':
            blob = torch.from_numpy(blob).to('cuda')
            with torch.no_grad():
                outputs = self.model(blob)
            net_out = [output.cpu().numpy() for output in outputs]
        elif self.backend == 'onnxruntime':
            net_out = self.model.run(self.output_names, {self.input_names[0]: blob})[0]
        else:
            print("unknown backend")
            net_out = None
        return net_out

    def release(self):
        if self.backend == 'onnxruntime':
            del self.model
        elif self.backend == 'tensorrt':
            del self.model.engine
            del self.model.context
            import gc
            gc.collect()
        else:
            pass
        self.model = None
    # def __call__(self, im_crops):
    #     embs = []
    #     for im in im_crops:
    #         if im is None:
    #             print("im is None")
    #             continue
    #         inp = preprocess(im)
    #         if self.backend=='tensorrt':
    #             inp = torch.from_numpy(inp).to('cuda')
    #             with torch.no_grad():
    #                 emb = self.model(inp)
    #             emb = emb.cpu().numpy()
    #         elif self.backend=='onnxruntime':
    #             emb = self.model.run(self.output_names, {self.input_names[0]: inp})[0]
    #         embs.append(emb.squeeze())
    #
    #     embs = np.array(np.stack(embs), dtype=np.float32)
    #     return embs