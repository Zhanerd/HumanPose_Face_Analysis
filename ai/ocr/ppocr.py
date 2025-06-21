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

    def detect(self, img, is_cls=False, det_thre=0.3, reg_thre=0.5, is_filter=False, min_area=0.01, min_slope=0.45):
        ori_im = img.copy()
        # 文字检测
        dt_boxes = self.det_model(img,det_thre)

        if dt_boxes is None:
            return None, None

        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        # 过滤倾斜的文本框和较小的文本框
        if is_filter:
            a = img.shape
            dt_boxes, area, angle = self.filter_tag_det_res(dt_boxes, a[0] * a[1], min_area, min_slope)

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
            if score >= reg_thre:
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

    def shoelace_area(self,points):
        """ 计算四边形的面积，points 是按顺时针或逆时针排列的 (x, y) 坐标列表 """
        points = points.astype(np.int32)
        n = len(points)
        area = 0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]  # 取下一个点，最后一个点连接到第一个点
            area += x1 * y2 - y1 * x2
        return int(abs(area) / 2)

    def slope(self, points):
        """ 计算两点连线与X轴的斜率 """
        x1, y1 = points[0]
        x2, y2 = points[1]
        if x2 - x1 == 0:  # 避免除以 0（垂直线的情况）
            return float('inf')  # 无穷大表示垂直
        return (y2 - y1) / (x2 - x1)

    def filter_tag_det_res(self, dt_boxes, image_area, ratio=0.01, min_slope=0.45):
        boxes = []
        areas = []
        angles = []
        for box in dt_boxes:
            area = self.shoelace_area(box)
            angle = self.slope(box)
            # print(f"面积: {area}, 总面积: {image_area}", angle)
            if area < image_area * ratio:
                continue
            elif abs(angle) > min_slope:
                continue
            else:
                boxes.append(box)
                areas.append(area)
                angles.append(angle)
            # angle = self.diagonal_slope(box)

            # rect = cv2.minAreaRect(box)
            # center, (width, height), angle = rect
            # if width < height:
            #     angle = 90 + angle

            # print(f"倾斜角度: {angle} 度")
            # if angle > 80:
            #     print("过滤掉倾斜角度大于80度的文本框")
            #     continue
            # else:
            #     boxes.append(box)
        return boxes,areas,angles


def is_invalid_ocr(text):
    # if len(text) != 3:
    #     return True
    if not text.isdigit():
        return True
    # if int(text) > 120:
    #     return True
    return False

if __name__ == "__main__":
    # det = TextDetector(model_path=r"D:\OnnxOCR-main\onnxocr\models\ppocrv4\det\det.onnx", gpu_id=0,model_input_size=(640,640),backend="onnxruntime")
    # img_path = r'D:\OnnxOCR-main\onnxocr\test_images\1.jpg'
    # img = cv2.imread(img_path)
    # result = det(img)
    # print(result)
    # ocr = PaddlePaddleOcr(gpu_id=0, det_path="/home/hz/hz_lyb/paddlepaddle/OnnxOCR-main/onnxocr/models/ppocrv4/det/det.engine", reg_path="/home/hz/hz_lyb/paddlepaddle/OnnxOCR-main/onnxocr/models/ppocrv4/rec/reg.engine",
    #                        cls_path="/home/hz/hz_lyb/paddlepaddle/OnnxOCR-main/onnxocr/models/ppocrv4/cls/cls.engine")
    ocr_det_model_path = r"D:\OnnxOCR-main\onnxocr\models\ppocrv4\det\det.onnx"
    ocr_reco_model_path = r"D:\OnnxOCR-main\onnxocr\models\ppocrv4\rec\rec.onnx"
    ocr_cls_model_path = r"D:\OnnxOCR-main\onnxocr\models\ppocrv4\cls\cls.onnx"
    ocr = PaddlePaddleOcr(gpu_id=0, det_path=ocr_det_model_path,
                          reg_path=ocr_reco_model_path,
                          cls_path=ocr_cls_model_path)
    # img_path = '/home/hz/hz_lyb/paddlepaddle/OnnxOCR-main/onnxocr/test_images/1.jpg'
    # img = cv2.imread(img_path)
    # result = ocr.detect(img,True)
    # print(result)
    cap = cv2.VideoCapture(r"D:\longrun_gushan\back_1_3.mp4")
    while True:
        ret, frame = cap.read()
        if ret:
            result = ocr.detect(frame,True)
            # print(result)
            for box, rec_result in zip(result[0], result[1]):
                text, score = rec_result
                if is_invalid_ocr(text):
                    continue
                print(text, score)
                cv2.polylines(frame, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                cv2.putText(frame, text, (int(box[0][0]), int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('image', cv2.resize(frame, (640, 480)))
        cv2.waitKey(0)