import copy
import numpy as np
import sys
import os
import cv2

sys.path.append('../../')
from ai.ocr.ppocr_det.ppocr_det import TextDetector
from ai.ocr.ppocr_reg.ppocr_reg import TextRecognizer
from ai.ocr.ppocr_cls.ppocr_cls import TextClassifier
from ai.ocr.utils import get_rotate_crop_image, get_minarea_rect_crop


class PaddlePaddleOcr():
    def __init__(self,gpu_id=0, det_path="", reg_path="", cls_path=""):
        """
        OCR工具类参数
        :param det_path: 文本检测模型路径
        :param reg_path: 文本特征提取模型路径
        :param gpu_id: <0为cpu,>=0为gpu
        :param cls_path: 文本分类模型路径，此处仅判断0度和180度的旋转文本
        """
        # 初始化模型
        self.gpu_id = gpu_id
        self.init_statue = False
        ### 识别的相关参数
        self.save_crop_res = False
        self.crop_image_res_index = 0
        self.drop_score = 0.5
        det_size = (640,640)
        # 加载文本检测模型
        self.det_model = None
        self.reg_model = None
        self.cls_model = None

        # 根据路径后缀判断模型类别
        if "onnx" in os.path.basename(det_path):
            det_backend = "onnxruntime"
        elif "engine" in os.path.basename(det_path):
            det_backend = "tensorrt"
        else:
            det_backend = "no support now"
        if "onnx" in os.path.basename(reg_path):
            reg_backend = "onnxruntime"
        elif "engine" in os.path.basename(reg_path):
            reg_backend = "tensorrt"
        else:
            reg_backend = "no support now"
        if "onnx" in os.path.basename(cls_path):
            cls_backend = "onnxruntime"
        elif "engine" in os.path.basename(cls_path):
            cls_backend = "tensorrt"
        else:
            cls_backend = "no support now"
        if os.path.exists(det_path) and det_path!='':
            self.det_model = TextDetector(model_path=det_path, gpu_id=gpu_id,backend=det_backend)
        else:
            print('det no success',det_path)
        if os.path.exists(reg_path) and reg_path!='':
            self.reg_model = TextRecognizer(model_path=reg_path, gpu_id=gpu_id,backend=reg_backend)
        else:
            print('reg no success',reg_path)
        if os.path.exists(cls_path) and cls_path!='':
            self.cls_model = TextClassifier(model_path=cls_path, gpu_id=gpu_id,backend=cls_backend)
        else:
            print('cls no success',cls_path)

        test = np.ones(shape=det_size, dtype=np.uint8)
        test = np.expand_dims(test, axis=-1)
        test = np.repeat(test, 3, axis=-1)
        # test = np.expand_dims(test, axis=-1)
        # print('test',test.shape)
        self.detect(test)

    def detect(self, img, is_cls=False):
        ori_im = img.copy()
        # 文字检测
        dt_boxes = self.det_model(img)

        if dt_boxes is None:
            return None, None

        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        # 图片裁剪
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.det_model.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # 方向分类
        if is_cls:
            img_crop_list, angle_list = self.cls_model(img_crop_list)

        # 图像识别
        rec_res = self.reg_model(img_crop_list)

        if self.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )

        self.crop_image_res_index += bbox_num

    def release(self):
        if self.det_model is not None:
            self.det_model.release()
        if self.reg_model is not None:
            self.reg_model.release()
        if self.cls_model is not None:
            self.cls_model.release()

if __name__ == "__main__":
    # det = TextDetector(model_path=r"D:\OnnxOCR-main\onnxocr\models\ppocrv4\det\det.onnx", gpu_id=0,model_input_size=(640,640),backend="onnxruntime")
    # img_path = r'D:\OnnxOCR-main\onnxocr\test_images\1.jpg'
    # img = cv2.imread(img_path)
    # result = det(img)
    # print(result)
    ocr = PaddlePaddleOcr(gpu_id=0, det_path="/home/hz/hz_lyb/paddlepaddle/OnnxOCR-main/onnxocr/models/ppocrv4/det/det.engine", reg_path="/home/hz/hz_lyb/paddlepaddle/OnnxOCR-main/onnxocr/models/ppocrv4/rec/reg.engine",
                           cls_path="/home/hz/hz_lyb/paddlepaddle/OnnxOCR-main/onnxocr/models/ppocrv4/cls/cls.engine")
    img_path = '/home/hz/hz_lyb/paddlepaddle/OnnxOCR-main/onnxocr/test_images/1.jpg'
    img = cv2.imread(img_path)
    result = ocr.detect(img,True)
    print(result)
