import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
from .db_postprocess import DBPostProcess
from ai.ocr.base import BaseTool
from ai.ocr.utils import transform, create_operators


class TextDetector(BaseTool):
    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (640, 640),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 gpu_id: int = 0,
                 backend: str = "tensorrt"):
        super().__init__(backend=backend, model_path=model_path, model_input_size=model_input_size, mean=mean, std=std,
                         gpu_id=gpu_id)
        self.det_box_type = "quad"
        self.det_thre = 0.3

        ###初始化前后处理类
        self.preprocess_op = None
        self.postprocess_op = None
        self.preprocess()
        self.postprocess()


    def __call__(self, img, det_thre):
        self.det_thre = det_thre
        ori_im = img.copy()
        data = {"image": img}
        # 前处理
        data = transform(data, self.preprocess_op)
        img, shape_list = data

        if img is None:
            return None, 0


        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        # 模型推理
        outputs = self.inference(img)

        preds = {}
        preds["maps"] = outputs[0]
        # 后处理
        result = self.postprocess_op(preds, shape_list)

        dt_boxes = result[0]["points"]

        if self.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        return dt_boxes

    def preprocess(self):
        pre_process_list = [
            {
                "DetResizeForTest": {
                    "limit_side_len": 640,
                    "limit_type": "max",
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        self.preprocess_op = create_operators(pre_process_list)

    def postprocess(self):
        postprocess_params = {}
        postprocess_params["name"] = "DBPostProcess"
        postprocess_params["thresh"] = self.det_thre
        postprocess_params["box_thresh"] = 0.6
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = "fast"
        postprocess_params["box_type"] = self.det_box_type
        self.postprocess_op = DBPostProcess(**postprocess_params)

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect


if __name__ == '__main__':
    det = TextDetector(r'D:\OnnxOCR-main\onnxocr\models\ppocrv4\det\det.onnx')
