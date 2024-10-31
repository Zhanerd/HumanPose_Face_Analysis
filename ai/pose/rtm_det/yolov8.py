from typing import List, Tuple
import torch
import torchvision

import cv2
import numpy as np

from ..base import BaseTool
from .post_processing import multiclass_nmsv2
class YOLOv8(BaseTool):

    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (640, 640),
                 nms_thr=0.45,
                 score_thr=0.7,
                 gpu_id: int = 0,
                 backend: str = "tensorrt"):
        super().__init__(model_path=model_path,backend=backend,
                         model_input_size=model_input_size,
                         gpu_id=gpu_id)
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.final_cls = list()

    ##### yolov8的调用看需求增加cls参数，默认只检测人，物体id要查coco_cat
    def __call__(self, image: np.ndarray,score_thr:float,cls:list = [0]):
        self.score_thr = score_thr
        if len(cls) == 0:
            self.final_cls = [0]
        else:
            self.final_cls = cls
        image, ratio = self.preprocess(image)
        image = np.expand_dims(image,axis=0)
        outputs = self.inference(image)[0]
        outputs = self.postprocess(outputs, ratio)
        return outputs

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(self.model_input_size[0] / img.shape[0],
                    self.model_input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img
        padded_img = padded_img / 255
        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMPose model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMPose model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        if outputs.ndim == 2 :
            outputs = np.expand_dims(outputs,axis=0)
        max_wh = 7680
        max_det = 300
        max_nms = 30000

        bs = outputs.shape[0]  # batch size
        nc = outputs.shape[1] - 4  # number of classes
        xc = np.amax(outputs[:, 4:4 + nc],1) > self.score_thr  # candidates

        output = [np.empty((0, 6))] * bs
        final_boxes = np.array([])
        final_scores = np.array([])
        final_cls_inds = np.array([])
        for index, x in enumerate(outputs):  # image index, image inference
            x = x.transpose(1, 0)[xc[index]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue
            box = x[:,:4]
            cls = x[:, 4:]

            box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
            if nc > 1:
                i, j = np.nonzero(cls > self.score_thr)
                x = np.concatenate((box[i.ravel(), :], x[i.ravel(), 4 + j.ravel(), None], j[:, None].astype(np.float32)), 1)
            else:  # best class only
                i, j = np.nonzero(cls > self.score_thr)
                x = np.concatenate((box[i.ravel(), :], x[i.ravel(), 4 + j.ravel(), None], j[:, None].astype(np.float32)), 1)
            if not x.shape[0]:  # no boxes
                continue
            sorted_idx = np.argsort(x[:, 4])[::-1][:max_nms]
            x = x[sorted_idx]
            # Batched NMS
            # c = x[:, 5:6] * max_wh  # classes
            # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            boxes, scores = x[:, :4], x[:, 4]
            s = x[:, 4:]
            boxes /= ratio
            dets = multiclass_nmsv2(boxes, s, self.nms_thr, self.score_thr)
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                isscore = final_scores > self.score_thr
                # iscat = final_cls_inds.astype(int) >= 0 ### 在这里的0即人的类别，参考coco或者其他数据集训练的标签序号增加
                iscat = list()
                for i in final_cls_inds:
                    iscat.append(i in self.final_cls)
                iscat = np.array(iscat)
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]

                final_boxes = final_boxes[isbbox]
                final_scores = final_scores[isbbox]
                final_cls_inds = final_cls_inds[isbbox]
                ### 过滤超出边界的框
                # filt = list(zip(final_boxes, final_scores))
                # fil_boxes = list()
                # fil_scores = list()
                # for box, score in filt:
                #     if sum(box) < 15000:
                #         fil_boxes.append(box)
                #         fil_scores.append(score)
                # final_boxes = np.array(fil_boxes)
                # final_scores = np.array(fil_scores)


                # results = np.concatenate((final_boxes, final_scores.reshape(-1, 1), final_cls_inds.reshape(-1, 1)), axis=1)
        # if self.final_cls==[0]:
        #     return final_boxes,final_scores
        # else:
        return final_boxes,final_scores,final_cls_inds
        # results = dict()
        # results['det'] = final_boxes
        # results['scores'] = final_scores
        # results['cls'] = final_cls_inds
        # return results

def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
