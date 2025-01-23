import numpy as np
import math
import cv2
from PIL import Image
import os

from ai.ocr.base import BaseTool
from .reg_postprocess import CTCLabelDecode


class TextRecognizer(BaseTool):
    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (3, 48, 320),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 gpu_id: int = 0,
                 batch_size: int = 16,
                 backend: str = "tensorrt"):
        super().__init__(backend=backend, model_path=model_path, model_input_size=model_input_size, mean=mean, std=std,
                         gpu_id=gpu_id)
        self.rec_batch_num = batch_size
        self.rec_image_shape = [v for v in self.model_input_size]
        # print('self.rec_image_shape',self.rec_image_shape,self.model_input_size)
        self.rec_algorithm = "SVTR_LCNet"

        ###初始化前后处理类
        self.preprocess_op = None
        self.postprocess_op = None
        self.preprocess()
        self.postprocess()


    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            # img = img[:, :, ::-1].transpose(2, 0, 1)
            # img = img[:, :, ::-1]
            # img = img.transpose(2, 0, 1)
            # img = img.astype(np.float32)
            # img = np.expand_dims(img, axis=0)
            # print(img.shape)
            outputs = self.inference(norm_img_batch)

            preds = outputs[0]

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res

    def preprocess(self):
        pass

    def postprocess(self):
        key_path = os.path.join(os.path.dirname(__file__), "ppocr_keys_v1.txt")
        # print(key_path)
        self.postprocess_op = CTCLabelDecode(
            character_dict_path=key_path,
            use_space_char=True,
        )

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        if self.rec_algorithm == "NRTR" or self.rec_algorithm == "ViTSTR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # return padding_im
            image_pil = Image.fromarray(np.uint8(img))
            if self.rec_algorithm == "ViTSTR":
                img = image_pil.resize([imgW, imgH], Image.BICUBIC)
            else:
                img = image_pil.resize([imgW, imgH], Image.ANTIALIAS)
            img = np.array(img)
            norm_img = np.expand_dims(img, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            if self.rec_algorithm == "ViTSTR":
                norm_img = norm_img.astype(np.float32) / 255.0
            else:
                norm_img = norm_img.astype(np.float32) / 128.0 - 1.0
            return norm_img
        elif self.rec_algorithm == "RFL":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
            resized_image = resized_image.astype("float32")
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
            resized_image -= 0.5
            resized_image /= 0.5
            return resized_image

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        # w = self.rec_onnx_session.get_inputs()[0].shape[3:][0]
        # w = self.rec_onnx_session.get_inputs()[0].shape[3:][0]
        # print(w)
        # if w is not None and w > 0:
        #     imgW = w

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        if self.rec_algorithm == "RARE":
            if resized_w > self.rec_image_shape[2]:
                resized_w = self.rec_image_shape[2]
            imgW = self.rec_image_shape[2]
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
