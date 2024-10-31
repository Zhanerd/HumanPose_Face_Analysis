import time
import sys
sys.path.append('../../')
from ai.pose import RTMDet, RTMPose, YOLOX, YOLOv8
from ai.pose.tracker.deep_sort import DeepSort

import cv2
import os

import numpy as np

def compute_iou(box1, box2):
    # box格式为[x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # 计算相交面积
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # 计算两个bbox的面积
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # 计算并集面积
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

class TopDownEstimation:
    def __init__(self, gpu_id=0, det_path="", pose_path="", track_path=""):
        """
        人体识别工具类
        :param gpu_id: 小于0则为CPU,大于等于0则为GPU
        :param det_path: 人体检测模型路径
        :param pose_path: 姿态估计模型路径
        """
        self.gpu_id = gpu_id
        self.init_statue = False

        # 加载人体检测模型
        self.det_model = None
        self.pose_model = None
        self.tracker = None
        det_size = (640,640)

        if "onnx" in det_path:
            det_backend = "onnxruntime"
        elif "engine" in det_path:
            det_backend = "tensorrt"
        else:
            det_backend = "no support now"
        if "onnx" in pose_path:
            pos_backend = "onnxruntime"
        elif "engine" in pose_path:
            pos_backend = "tensorrt"
        else:
            pos_backend = "no support now"
        if os.path.exists(det_path) and os.path.exists(pose_path):
            if "yolo" in det_path:
                if 'yolox' in det_path:
                    if "tiny" in det_path:
                        det_size = (416,416)
                    else:
                        det_size = (640,640)
                    self.det_model = YOLOX(model_path=det_path, gpu_id=gpu_id,model_input_size=det_size,backend=det_backend)
                elif 'yolov8' in det_path:
                    det_size = (640, 640)
                    self.det_model = YOLOv8(model_path=det_path, gpu_id=gpu_id,model_input_size=det_size,backend=det_backend)
            else:
                if "hand" in det_path or 'end' in det_path:
                    det_size = (320, 320)
                else:
                    det_size = (640, 640)
                self.det_model = RTMDet(model_path=det_path, gpu_id=gpu_id,model_input_size=det_size,backend=det_backend)

            if "hand" in pose_path:
                in_size = (256,256)
            elif "l_wholebody" in pose_path:
                in_size = (288,384)
            else:
                in_size = (192,256)

            self.pose_model = RTMPose(model_path=pose_path, gpu_id=gpu_id,model_input_size=in_size,backend=pos_backend)
            self.init_statue = True
        else:
            print(det_path, pose_path)
        if track_path != "" and os.path.exists(track_path):
            if "onnx" in track_path:
                tra_backend = "onnxruntime"
            elif "engine" in track_path:
                tra_backend = "tensorrt"
            else:
                tra_backend = "no support now"
            self.tracker = DeepSort(model_path=track_path, gpu_id=gpu_id,backend=tra_backend)

        test = np.ones(shape=det_size, dtype=np.uint8)
        test = np.expand_dims(test, axis=-1)
        test = np.repeat(test, 3, axis=-1)
        self.estimate(test, 0.5)
    def track(self, img, bbox):
        if self.tracker is None:
            print("没有加载追踪模型")
            return []
        else:
            ##### 注意！！！，进去的bbox和出来的bbox顺序不一定一致，
            ##### 且位置有些偏差，通过iou过滤（用compute_iou函数），建议取95为阈值
            track_bbox, ids = self.tracker.update(bbox, img)
            return track_bbox,ids
            # tracks = self.tracker.update(bbox, img)
            # return tracks

    def det_es(self, img, score_thr,cls=[0]):
        results = dict()
        if type(self.det_model).__name__=='YOLOv8':
            dets, scores, clses = self.det_model(img, score_thr=score_thr,cls=cls)
            results['det'] = dets
            results['scores'] = scores
            results['cls'] = clses
            return results
        else:
            dets, scores = self.det_model(img, score_thr=score_thr)
            results['det'] = dets
            results['scores'] = scores
            return results
        # results = self.det_model(img, score_thr=score_thr, cls=cls)


    def pose_es(self, img, bbox):
        ### 取追踪结果的框
        for b in bbox:
            if len(b)>4:
                b = b[:4]
        keypoints, scores = self.pose_model(img,bbox)
        return keypoints, scores

    def estimate(self, img, score_thr, track=False):
        results = list()
        if not self.init_statue:
            return results
        det_time = time.time()
        if type(self.det_model).__name__=='YOLOv8':
            det, det_score,cls = self.det_model(img, score_thr=score_thr)
            det = [box for i,box in enumerate(det) if cls[i] == 0]
        else:
            det, det_score = self.det_model(img, score_thr=score_thr)
        print("det time:", time.time() - det_time)
        # det_results = self.det_model(img, score_thr=score_thr)
        # det = det_results['det']
        # det_score = det_results['scores']
        track_ids = []
        track_time = time.time()
        if track and self.tracker is not None:
            det, track_ids = self.track(img, det)
        print("track time:", time.time() - track_time)
        kps_time = time.time()
        kps, kps_score = self.pose_model(img, det)
        print("kps time:", time.time() - kps_time)
        det_score = det_score[:len(det)]
        for i in range(len(det)):
            result = dict()
            result['det'] = det[i]
            if not track:
                result['det_score'] = det_score[i]
            result['kps'] = kps[i]
            result['kps_score'] = kps_score[i]
            result['track_id'] = 0
            if track and self.tracker is not None:
                result['track_id'] = track_ids[i]
            results.append(result)
        return results


    def release(self):
        self.det_model.release()
        self.pose_model.release()

if __name__ == '__main__':
    pose = TopDownEstimation(det_path=r'D:\ai\ai\models\yolov8_m.engine',
                             pose_path=r'D:\ai\ai\models\rtmpose_m_halpe262.engine',
                             track_path=r'D:\ai\ai\models\deepsort.engine')
    # cap = cv2.VideoCapture(0)
    runway_info = {
        "1": [[28, 900], [308, 900], [636, 990], [90, 1024]],
        "2": [[308, 900], [776, 868], [1148, 952], [636, 990]],
        "3": [[776, 868], [1314, 850], [1632, 914], [1148, 952]],
        "4": [[1314, 850], [1766, 824], [2086, 880], [1632, 914]],
        "5": [[1766, 824], [2170, 796], [2520, 850], [2086, 880]]
    }
    cap = cv2.VideoCapture(r'D:\replay\10月22日(1).mp4')
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        t1 = time.time()
        # dets,scores,cls = pose.det_es(frame, 0.3, list(range(81)))
        ### 椅子56，人0，
        results = pose.estimate(frame, 0.3, True)
        # boxes = []
        # for result in results:
        #     boxes.append(result['det'])
        #     print(result['track_id'])
        # l = pose.track(frame, boxes)
        # print('len',len(dets))
        # for i,box in enumerate(dets):
        #     frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0),3)
        #     frame = cv2.putText(frame, str(cls[i]),  (int(box[0]), int(box[1])),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        for result in results:
            frame = cv2.rectangle(frame, (int(result['det'][0]), int(result['det'][1])), (int(result['det'][2]), int(result['det'][3])), (0, 255, 0),3)
            frame = cv2.putText(frame, str(result['track_id']), (int(result['det'][0]), int(result['det'][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for ids,kps in enumerate(result['kps']):
                frame = cv2.circle(frame, (int(kps[0]), int(kps[1])), 3, (0, 0, 255), -1)
                frame = cv2.putText(frame, str(ids), (int(kps[0]), int(kps[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 222, 255),
                                    2)
        for rw_id, runway in runway_info.items():
            runway = np.array(runway, dtype=np.int32)
            for i,run in enumerate(runway):
                runway[i] =(int(run[0]*(1920/2560)),(run[1]*(1080/1440)))
            frame = cv2.polylines(frame, [runway], True, (0, 255, 0), 2)
            frame = cv2.putText(frame, str(rw_id), (int(runway[0][0]),int(runway[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 3)

        # for i in l:
        #     frame = cv2.putText(frame, str(i[4]), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        #print(len(results))
