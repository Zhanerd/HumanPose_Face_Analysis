import onnxruntime
import tensorrt as trt
import os

from ai.torch2trt import TRTModule
from ai.face_recognition.face import Face
from ai.face_recognition.face_det.retinaface import RetinaFace
from ai.face_recognition.face_reg.arcface import ArcFace


class FaceAnalysis:
    def __init__(self, det_path='', reg_path=''):
        if 'onnx' in os.path.basename(det_path):
            print('loading det onnx model')
            self.det_session = onnxruntime.InferenceSession(det_path, providers=['CUDAExecutionProvider'])
            print(self.det_session.get_providers())
        elif 'engine' in os.path.basename(det_path):
            det_logger = trt.Logger(trt.Logger.INFO)
            with open(det_path, 'rb') as f, trt.Runtime(det_logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            det_input = []
            det_output = []
            if trt.__version__.split('.')[0] == '10':
                # rt10.x
                for name in engine:
                    mode = engine.get_tensor_mode(name)
                    if mode == trt.TensorIOMode.INPUT:
                        det_input.append(name)
                    elif mode == trt.TensorIOMode.OUTPUT:
                        det_output.append(name)
            else:
                # rt8.x
                for i in range(engine.num_bindings):
                    if engine.binding_is_input(i):
                        det_input.append(engine.get_binding_name(i))
                    else:
                        det_output.append(engine.get_binding_name(i))

            self.det_session = TRTModule(engine, input_names=det_input, output_names=det_output)
        else:
            print('no support now')
        if 'onnx' in os.path.basename(reg_path):
            print('loading reg onnx model')
            self.reg_session = onnxruntime.InferenceSession(reg_path, providers=['CUDAExecutionProvider'])
            print(self.reg_session.get_providers())
        elif 'engine' in os.path.basename(reg_path):
            reg_logger = trt.Logger(trt.Logger.INFO)
            with open(reg_path, 'rb') as f, trt.Runtime(reg_logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            reg_input = []
            reg_output = []
            if trt.__version__.split('.')[0] == '10':
                # rt10.x
                for name in engine:
                    mode = engine.get_tensor_mode(name)
                    if mode == trt.TensorIOMode.INPUT:
                        reg_input.append(name)
                    elif mode == trt.TensorIOMode.OUTPUT:
                        reg_output.append(name)
            else:
                # rt8.x
                for i in range(engine.num_bindings):
                    if engine.binding_is_input(i):
                        reg_input.append(engine.get_binding_name(i))
                    else:
                        reg_output.append(engine.get_binding_name(i))

            self.reg_session = TRTModule(engine, input_names=reg_input, output_names=reg_output)
        else:
            print('no support now')
        self.det_model = RetinaFace(model_file=det_path, session=self.det_session)
        self.reg_model = ArcFace(model_file=reg_path, session=self.reg_session)

        self.faces_queue = []

    def release(self):
        self.det_model.release()
        self.reg_model.release()

    def prepare(self, gpu_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        self.det_model.prepare(gpu_id, input_size=det_size, det_thresh=det_thresh)
        self.reg_model.prepare(gpu_id)

    ### 一次处理一张人脸
    def get(self, img, det_thresh=0.5, nms_thr=None, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default',
                                             det_thresh=det_thresh,
                                             nms_thr=nms_thr)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            self.reg_model.get(img, face)
            ret.append(face)
        return ret

    ### 一次处理多张人脸(增加模型吞吐量)
    def batch_get(self, img, det_thresh=0.5, nms_thr=None, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default',
                                             det_thresh=det_thresh,
                                             nms_thr=nms_thr)
        if bboxes.shape[0] == 0:
            return []
        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            faces.append(face)
        ret = self.reg_model.batch_get(img, faces)
        return ret

    ### 等待人脸满足一定的batch数量，再进行特征提取
    def batch_get_queue(self, img, det_thresh=0.5, nms_thr=None, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default',
                                             det_thresh=det_thresh,
                                             nms_thr=nms_thr)
        if bboxes.shape[0] == 0:
            return []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            self.faces_queue.append(face)
        if len(self.faces_queue) < 32:  ### 32 张人脸才进行特征提取
            return []
        elif len(self.faces_queue) > 64:  ### 64 是模型最大batchsize
            ret = self.reg_model.batch_get(img, self.faces_queue[:64])
            self.faces_queue = self.faces_queue[64:]
            return ret
        else:
            ret = self.reg_model.batch_get(img, self.faces_queue)
            self.faces_queue = []
            return ret
