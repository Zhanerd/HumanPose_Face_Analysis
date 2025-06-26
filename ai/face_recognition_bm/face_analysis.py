import glob
import os.path as osp
import sophon.sail as sail

from .face import Face
from .retinaface import RetinaFace
from .arcface import ArcFace

class FaceAnalysis:
    def __init__(self, det_bmodel: str, reg_bmodel: str, device_id: int = 0):
        # 1) RetinaFace SAIL Engine & Tensor
        self.det = RetinaFace(det_bmodel, device_id)
        # 2) ArcFace  SAIL Engine & Tensor
        self.reg = ArcFace(reg_bmodel, device_id)

    def release(self):
        del self.det_session
        del self.reg_session

    def prepare(self,
                det_size: tuple[int,int] = (640, 640),
                det_thresh: float       = 0.5):
        """初始化推理输入大小和阈值"""
        self.det.prepare(input_size=det_size, det_thresh=det_thresh)

    def get(self, img, det_thresh=0.5, nms_thr=0.2, max_num=0, need_feature=True):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default',
                                             det_thresh=det_thresh,
                                             nms_thr=nms_thr)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            if need_feature:
                self.reg_model.get(img, face)
            ret.append(face)
        return ret