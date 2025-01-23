import cv2
import math
import numpy as np
import onnxruntime

input_shape = (640, 640)
score_threshold = 0.4
nms_threshold = 0.4
confidence_threshold = 0.4


def nms(boxes, scores, score_threshold, nms_threshold):
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


def filter_box(outputs,nc=3):  # 过滤掉无用的框
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
        if score > score_threshold:
            rotated_boxes.append(np.concatenate([outputs[:4, i], np.array([score, class_id, angle * 180 / math.pi])]))
            scores.append(score)
            class_ids.append(class_id)

    rotated_boxes = np.array(rotated_boxes)
    boxes = xywh2xyxy(rotated_boxes)
    scores = np.array(scores)
    indices = nms(boxes, scores, score_threshold, nms_threshold)
    output = rotated_boxes[indices]
    return output


def letterbox(im, new_shape=(416, 416), color=(114, 114, 114)):
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
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def scale_boxes(boxes, shape):
    # Rescale boxes (xyxy) from input_shape to shape
    gain = min(input_shape[0] / shape[0], input_shape[1] / shape[1])  # gain  = old / new
    pad = (input_shape[1] - shape[1] * gain) / 2, (input_shape[0] - shape[0] * gain) / 2  # wh padding
    boxes[..., [0, 1]] -= pad  # xy padding
    boxes[..., :4] /= gain
    return boxes


def draw(image, box_data):
    box_data = scale_boxes(box_data, image.shape)
    boxes = box_data[..., :4]
    scores = box_data[..., 4]
    classes = box_data[..., 5].astype(np.int32)
    angles = box_data[..., 6]
    for box, score, cl, angle in zip(boxes, scores, classes, angles):
        rotate_box = ((box[0], box[1]), (box[2], box[3]), angle)
        points = cv2.boxPoints(rotate_box)
        points = np.int0(points)
        cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=1)
        cv2.putText(image, '{0} {1:.2f}'.format(cl, score), (points[0][0], points[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


import math







if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\84728\Desktop\2ball_cz_1.jpg", -1)
    input = letterbox(image, input_shape)
    input = input[:, :, ::-1].transpose(2, 0, 1).astype(dtype=np.float32)  # BGR2RGB和HWC2CHW
    input = input / 255.0
    input_tensor = []
    input_tensor.append(input)

    onnx_session = onnxruntime.InferenceSession(r"C:\Users\84728\Desktop\best.onnx",
                                                providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])

    input_name = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)

    output_name = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)

    inputs = {}
    for name in input_name:
        inputs[name] = np.array(input_tensor)

    outputs = onnx_session.run(None, inputs)[0]
    print(outputs.shape)

    boxes = filter_box(outputs)
    draw(image, boxes)
    cv2.imwrite('result.jpg', image)
