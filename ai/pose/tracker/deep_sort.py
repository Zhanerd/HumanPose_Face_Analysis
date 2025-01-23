import time

import numpy as np
#import torch

from .feature_extractor import Extractor
# from .extractor import Extractor
from .nn_matching import NearestNeighborDistanceMetric
from .detection import Detection
from .tracker import Tracker

class DeepSort(object):
    def __init__(self, model_path, max_dist=0.4, min_confidence=0.2, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, gpu_id=0):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path,gpu_id)

        max_cosine_distance = max_dist
        nn_budget = nn_budget
        metric = NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)     #euclidean，cosine

        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xyxy, ori_img):
        if len(bbox_xyxy) == 0:
            return [],[]
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xyxy, ori_img)
        bbox_tlwh = self._xyxy_to_tlwh(np.array(bbox_xyxy))
        detections = [Detection(bbox_tlwh[i], features[i]) for i, box in enumerate(bbox_xyxy)]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities

        output_bbox = []
        output_id = []

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            output_bbox.append([x1,y1,x2,y2])
            output_id.append(track_id)
            # outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int32))
        # if len(outputs) > 0:
        #     outputs = np.stack(outputs,axis=0)
        # return outputs
        return output_bbox,output_id


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        # elif isinstance(bbox_xywh, torch.Tensor):
        #     bbox_tlwh = bbox_xywh.clone()
        if len(bbox_xywh) == 0:
            return []
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        # 计算边界框的宽度和高度
        width = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        height = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]

        # 构建新的边界框坐标数组
        tlwh_bbox = np.column_stack((bbox_xyxy[:, 0], bbox_xyxy[:, 1], width, height))

        # 返回转换后的坐标格式
        return tlwh_bbox
    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        for box in bbox_xyxy:
            box = [int(max(0, item)) for item in box]
            im = ori_img[box[1]:box[3],box[0]:box[2]]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


