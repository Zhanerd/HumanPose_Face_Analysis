from typing import List, Tuple
import torch

import cv2
import numpy as np

from ai.pose.base import BaseTool
from ai.pose.rtm_det.post_processing import multiclass_nmsv2


class YOLOv8(BaseTool):
    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (640, 640),
                 nms_thr=0.45,
                 score_thr=0.7,
                 gpu_id: int = 0):
        super().__init__(model_path=model_path,
                         model_input_size=model_input_size,
                         gpu_id=gpu_id)
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.final_cls = list()

    ##### yolov8的调用看需求增加cls参数，默认只检测人，物体id要查coco_cat
    def __call__(self, image: np.ndarray, score_thr: float, cls: list = [0]):
        self.score_thr = score_thr
        if len(cls) == 0:
            self.final_cls = [0]
        else:
            self.final_cls = cls
        image, ratio = self.preprocess(image)
        image = np.expand_dims(image, axis=0)
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
        if outputs.ndim == 2:
            outputs = np.expand_dims(outputs, axis=0)
        max_wh = 7680
        max_det = 300
        max_nms = 30000

        bs = outputs.shape[0]  # batch size
        nc = outputs.shape[1] - 4  # number of classes
        xc = np.amax(outputs[:, 4:4 + nc], 1) > self.score_thr  # candidates

        output = [np.empty((0, 6))] * bs
        final_boxes = np.array([])
        final_scores = np.array([])
        final_cls_inds = np.array([])
        for index, x in enumerate(outputs):  # image index, image inference
            x = x.transpose(1, 0)[xc[index]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue
            box = x[:, :4]
            cls = x[:, 4:]

            box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
            if nc > 1:
                i, j = np.nonzero(cls > self.score_thr)
                x = np.concatenate(
                    (box[i.ravel(), :], x[i.ravel(), 4 + j.ravel(), None], j[:, None].astype(np.float32)), 1)
            else:  # best class only
                i, j = np.nonzero(cls > self.score_thr)
                x = np.concatenate(
                    (box[i.ravel(), :], x[i.ravel(), 4 + j.ravel(), None], j[:, None].astype(np.float32)), 1)
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
                final_cls_inds = final_cls_inds.astype(np.int8)
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
        return final_boxes, final_scores, final_cls_inds
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


import time

if __name__ == '__main__':
    # model = YOLOv8(model_path=r"D:\ai_library\ai\models\yolov8_m.onnx")
    # model = YOLOv8(model_path=r"D:\yolov8_m2.onnx")
    model = YOLOv8(model_path=r"C:\Users\84728\Desktop\m_ballaug_v1.2.onnx")
    ball_model = YOLOv8(model_path=r"D:\ai_library\ai\models\b1.onnx", model_input_size=(320, 320))
    video_path = r"C:\Users\84728\Desktop\802125.mp4"
    video_path = r"D:\篮球\篮球\192.168.1.100球出界.mp4"
    video_path = r"D:\篮球\long_1957979962486493185.mp4"
    # video_path = r"C:\Users\84728\Desktop\longjump_0610\StandingLongJump_1837069828510756866_1749521271.mp4"
    #video_path = r"C:\Users\84728\Documents\WeChat Files\wxid_z1yd99twbh4712\FileStorage\File\2025-07\Volleyball\111136_200211_9_172004_971.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    # cls = list(range(80))
    t_size = 320
    while cap.isOpened():
        success, frame = cap.read()
        # # 应用高斯模糊减少噪音
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
        #
        # edges = cv2.Canny(blurred_image, threshold1=10, threshold2=100)
        #
        # # 使用霍夫变换检测圆形
        # circles = cv2.HoughCircles(
        #     edges,
        #     cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100
        # )
        #
        # # 检测到圆形时
        # if circles is not None:
        #     circles = np.round(circles[0, :]).astype("int")
        #     for (x, y, r) in circles:
        #         # 绘制圆形
        #         cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        #         # 绘制圆心
        #         cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # frame = frame[:540, :910, :]
        if not success:
            continue
        h, w, _ = frame.shape
        detect_img = frame.copy()
        ### 椅子56，人0，球32.
        t1 = time.time()
        boxes, scores, cls_inds = model(frame, 0.1, [0, 2])
        print('inference', time.time() - t1)
        # print(cls_inds)
        for box, score, cls in zip(boxes, scores, cls_inds):
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, str(cls) + str(round(score, 2)), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            if cls == 0:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                # 根据中心点裁剪区域（限制在图像范围内）
                src_y1 = max(int(cy - t_size / 2), 0)
                src_y2 = min(int(cy + t_size / 2), h)
                src_x1 = max(int(cx - t_size / 2), 0)
                src_x2 = min(int(cx + t_size / 2), w)
                detect_img = frame.copy()
                imgs = frame - detect_img
                # ball_boxes,_,_ = ball_model(detect_img, 0.3)
                # for ball_box in ball_boxes:
                #     print(len(ball_box),ball_box)
                #     cv2.rectangle(detect_img, (int(ball_box[0]), int(ball_box[1])), (int(ball_box[2]), int(ball_box[3])), (0, 255, 0), 2)
        # box.astype(int)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        # cv2.imshow('edges', imgs)
        # cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
