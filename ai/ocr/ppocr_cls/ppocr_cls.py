import cv2
import copy
import numpy as np
import math

from ai.ocr.base import BaseTool
from .cls_postprocess import ClsPostProcess

class TextClassifier(BaseTool):
    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (3, 48, 192),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 gpu_id: int = 0,
                 batch_size: int = 6,
                 backend: str = "tensorrt"):
        super().__init__(backend=backend, model_path=model_path, model_input_size=model_input_size, mean=mean, std=std,
                         gpu_id=gpu_id)
        self.cls_image_shape = [v for v in self.model_input_size]
        self.cls_batch_num = batch_size
        self.cls_thresh = 0.9

        ###初始化前后处理类
        self.preprocess_op = None
        self.postprocess_op = None
        self.preprocess()
        self.postprocess()


    def __call__(self, img_list):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [["", 0.0]] * img_num
        batch_num = self.cls_batch_num

        for beg_img_no in range(0, img_num, batch_num):

            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0

            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            outputs = self.inference(norm_img_batch)
            
            prob_out = outputs[0]

            cls_result = self.postprocess_op(prob_out)
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if "180" in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )
        return img_list, cls_res

    def preprocess(self):
        pass

    def postprocess(self):
        ### ["0", "180"] 目前仅支持反转
        self.postprocess_op = ClsPostProcess(label_list=["0", "180"])

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        if self.cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


