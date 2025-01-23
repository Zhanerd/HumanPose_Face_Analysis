from typing import List, Tuple
import torch
import math
import cv2
import numpy as np

from base import BaseTool


def nms(boxes, scores, nms_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= nms_threshold)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


class YOLO_OBB(BaseTool):
    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (640, 640),
                 nms_thr=0.45,
                 score_thr=0.7,
                 gpu_id: int = 0):
        """
        YOLO_OBB工具类参数
        :param model_path: obb模型路径
        :param model_input_size: 模型输入大小
        :param gpu_id: <0为cpu,>=0为gpu
        :param score_thr: 检测阈值
        :param nms_thr: nms阈值
        """
        super().__init__(model_path=model_path,
                         model_input_size=model_input_size,
                         gpu_id=gpu_id)
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.final_cls = list()

    def __call__(self, image: np.ndarray, score_thr: float):
        self.score_thr = score_thr
        pre_image = self.preprocess(image)
        outputs = self.inference(pre_image)[0]
        ### 此时的输出未经过缩放
        outputs = self.postprocess(outputs)
        results = self.format_results(image.shape, outputs)
        return results

    def preprocess(self, im: np.ndarray, new_shape: tuple = (640, 640), color: tuple = (114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        input = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  # BGR2RGB和HWC2CHW
        input = input / 255.0
        return input

    def postprocess(self, outputs, nc=3):  ### nc 为类别数
        outputs = np.squeeze(outputs)

        rotated_boxes = []
        scores = []
        class_ids = []
        classes_scores = outputs[4:(4 + nc), ...]
        angles = outputs[-1, ...]

        for i in range(outputs.shape[1]):
            class_id = np.argmax(classes_scores[..., i])
            score = classes_scores[class_id][i]
            angle = angles[i]
            if 0.5 * math.pi <= angle <= 0.75 * math.pi:
                angle -= math.pi
            if score > self.score_thr:
                rotated_boxes.append(
                    np.concatenate([outputs[:4, i], np.array([score, class_id, angle * 180 / math.pi])]))
                scores.append(score)
                class_ids.append(class_id)

        rotated_boxes = np.array(rotated_boxes)
        boxes = xywh2xyxy(rotated_boxes)
        scores = np.array(scores)
        indices = nms(boxes, scores, self.nms_thr)
        output = rotated_boxes[indices]
        return output

    def box_points(self, center, size, angle_deg):
        """
        center: (cx, cy)，旋转矩形中心坐标
        size: (w, h)，旋转矩形的宽和高
        angle_deg: 旋转角度，单位是度 (degrees)，
                   其定义与 OpenCV 保持一致（逆时针为正）。
        返回：4 个顶点的坐标 [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        """
        cx, cy = center
        w, h = size
        # 将角度从度转换为弧度
        # 将角度从度转换为弧度
        angle_rad = math.radians(angle_deg)

        # 对应源码中的 a 和 b
        # 注意源码里 a=sin(angle)*0.5, b=cos(angle)*0.5
        a = math.sin(angle_rad) * 0.5
        b = math.cos(angle_rad) * 0.5

        # 计算 4 个顶点
        # 下列计算与 C++ 源码一致
        # pt0 = np.array([cx - a * h - b * w, cy + b * h - a * w])
        # pt1 = np.array([cx + a * h - b * w, cy - b * h - a * w])
        # pt2 = np.array([cx + a * h + b * w, cy - b * h + a * w])
        # pt3 = np.array([cx - a * h + b * w, cy + b * h + a * w])
        ### 不低于0
        pt0 = np.array([max(cx - a * h - b * w, 0), max(cy + b * h - a * w,0 )])
        pt1 = np.array([max(cx + a * h - b * w, 0), max(cy - b * h - a * w, 0)])
        pt2 = np.array([max(cx + a * h + b * w, 0), max(cy - b * h + a * w, 0)])
        pt3 = np.array([max(cx - a * h + b * w, 0), max(cy + b * h + a * w, 0)])

        return np.stack([pt0, pt1, pt2, pt3], axis=0)

    def scale_boxes(self, boxes, shape):
        # Rescale boxes (xyxy) from input_shape to shape
        gain = min(self.model_input_size[0] / shape[0], self.model_input_size[1] / shape[1])  # gain  = old / new
        pad = (self.model_input_size[1] - shape[1] * gain) / 2, (
                    self.model_input_size[0] - shape[0] * gain) / 2  # wh padding
        boxes[..., [0, 1]] -= pad  # xy padding
        boxes[..., :4] /= gain
        return boxes

    def format_results(self, shape, outputs):
        box_data = self.scale_boxes(outputs, shape)
        boxes = box_data[..., :4]
        bboxes = xywh2xyxy(boxes)
        scores = box_data[..., 4]
        classes = box_data[..., 5].astype(np.int32)
        angles = box_data[..., 6]
        results = []
        for box, score, cl, angle, bbox in zip(boxes, scores, classes, angles, bboxes):
            result = {}
            points = self.box_points((box[0], box[1]), (box[2], box[3]), angle)
            points = np.int0(points)
            result['points'] = points
            result['score'] = score
            result['class'] = cl
            result['angle'] = angle
            result['det'] = bbox
            results.append(result)
        return results

if __name__ == '__main__':
    model = YOLO_OBB(model_path=r"C:\Users\84728\Desktop\best.engine",
                     model_input_size=(640, 640),
                     nms_thr=0.45,
                     score_thr=0.7,
                     gpu_id=0)

    img = cv2.imread(r"C:\Users\84728\Desktop\2ball_cz_1.jpg")
    results = model(img,0.4)
    for result in results:
        points = np.int0(result["points"])
        cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=1)
        cv2.putText(img, '{0} {1:.2f}'.format(result['class'], result['score']), (points[0][0], points[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    print(results)
