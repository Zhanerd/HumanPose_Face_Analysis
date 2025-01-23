import os.path
import time

import onnxruntime
import numpy as np
import cv2
import torch


# 特征检测和质量判断是两个分支，先训人脸特征后，frozen人脸分支，在训质量分支，对嘴巴区域遮挡差
class FaceQuality:
    def __init__(self, backbone_path, quality_path, gpu_id=0):
        self.gpu_id = gpu_id
        if gpu_id >= 0:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.backbone = onnxruntime.InferenceSession(backbone_path, providers=providers)
        self.quality = onnxruntime.InferenceSession(quality_path, providers=providers)

    def inference(self, img):
        # 人脸图片初始化
        resized = cv2.resize(img, (112, 112))
        ccropped = resized[..., ::-1]  # BGR to RGB
        ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
        ccropped = np.reshape(ccropped, [1, 3, 112, 112])
        ccropped = np.array(ccropped, dtype=np.float32)
        ccropped = (ccropped - 127.5) / 128.0

        # backbone推理获得初步人脸特征向量
        input_backbone = self.backbone.get_inputs()
        input_data = {input_backbone[0].name: ccropped}
        outputs = self.backbone.get_outputs()
        output_backbone = []
        for o in outputs:
            output_backbone.append(o.name)
        _, fc = self.backbone.run(output_backbone, input_data)
        # quality推理获得人脸质量分数
        input_details2 = self.quality.get_inputs()
        input_data2 = {input_details2[0].name: fc}

        outputs2 = self.quality.get_outputs()
        output_names2 = []
        for o in outputs2:
            output_names2.append(o.name)

        score = self.quality.run(output_names2, input_data2)

        return score[0]

class face_quality_assessment:
    def __init__(self,quality_path, gpu_id=0):
        self.gpu_id = gpu_id
        self.input_height = 112
        self.input_width = 112
        if 'onnx' in os.path.basename(quality_path):
            if gpu_id >= 0:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            self.quality = onnxruntime.InferenceSession(quality_path, providers=providers)
            self.backend = 'onnxruntime'
        elif 'engine' in os.path.basename(quality_path):
            import tensorrt as trt
            from ai.torch2trt import TRTModule
            logger = trt.Logger(trt.Logger.INFO)
            with open(quality_path, 'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())

            input_names = []
            output_names = []
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i):
                    input_names.append(engine.get_binding_name(i))
                else:
                    output_names.append(engine.get_binding_name(i))
            self.session = TRTModule(engine,input_names=input_names, output_names=output_names)
            self.backend = 'tensorrt'

    def inference(self, srcimg):
        input_img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        input_img = (input_img.astype(np.float32) / 255.0 - 0.5) / 0.5
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
        if self.backend == 'onnxruntime':
            input_name = self.quality.get_inputs()[0].name
            scores = self.quality.run(None, {input_name:input_img})
        elif self.backend == 'tensorrt':
            input = torch.from_numpy(input_img).to('cuda')
            with torch.no_grad():
                outputs = self.session(input)
            outputs = [output.cpu().numpy() for output in outputs]
            scores = outputs[0]
        return scores

# 不同层人脸特征的输出，通过对比输出的一致性判断人脸质量高低，对遮挡反应强，模型轻，但是模糊识别差
class face_quality_assessment_cv2dnn():
    def __init__(self, path):
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 112
        self.input_width = 112

    def detect(self, srcimg):
        input_img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        input_img = (input_img.astype(np.float32) / 255.0 - 0.5) / 0.5

        blob = cv2.dnn.blobFromImage(input_img.astype(np.float32))
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outputs[0].reshape(-1)


if __name__ == '__main__':
    from ai.face_recognition.face_reco import FaceRecognition

    # face_quality = FaceQuality('ai/models/quality_backbone.onnx', 'ai/models/face_quality.onnx')
    fq = face_quality_assessment(r'C:\Users\84728\PycharmProjects\face\ai\models\face_quality_assessment.engine',gpu_id=0)
    img = cv2.imread(r"C:\Users\84728\Desktop\fq1.jpg")
    face_det = FaceRecognition(det_path=r'C:\Users\84728\PycharmProjects\face\ai\models\face_det_10g.engine', reg_path=r'C:\Users\84728\PycharmProjects\face\ai\models\face_w600k_r50.engine',gpu_id=-1)
    image = img.copy()
    t1 = time.time()
    results = face_det.detect(img)
    for result in results:
        image = cv2.rectangle(image, (result['bbox'][0], result['bbox'][1]), (result['bbox'][2], result['bbox'][3]),
                              (0, 0, 255), 2)
        image = image[result['bbox'][1]:result['bbox'][3], result['bbox'][0]:result['bbox'][2]]
    #result = face_quality.inference(image)
    s = fq.inference(image)
    print(s)
    print(time.time() - t1)