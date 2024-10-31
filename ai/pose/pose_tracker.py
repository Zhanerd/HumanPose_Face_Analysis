import warnings

import numpy as np

from ai.pose.mmpose import TopDownEstimation
from skeleton import *  # noqa

def compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn('union_area=0 is unexpected')

    iou = inter_area / union_area

    return iou


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Get bounding box from keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of person.
        expansion (float): Expansion ratio of bounding box.

    Returns:
        np.ndarray: Bounding box of person.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox


class PoseTracker:
    MIN_AREA = 1000

    def __init__(self,
                 det_path: str,
                 pose_path: str,
                 det_frequency: int = 1,
                 tracking: bool = True,
                 tracking_thr: float = 0.3):

        model = TopDownEstimation(det_path=det_path, pose_path=pose_path)

        self.det_model = model.det_model
        self.pose_model = model.pose_model

        self.det_frequency = det_frequency
        self.tracking = tracking
        self.tracking_thr = tracking_thr
        self.reset()

        if self.tracking:
            print('Tracking is on, you can get higher FPS by turning it off:'
                  '`PoseTracker(tracking=False)`')

    def reset(self):
        """Reset pose tracker."""
        self.frame_cnt = 0
        self.next_id = 0
        self.bboxes_last_frame = []
        self.track_ids_last_frame = []

    def __call__(self, image: np.ndarray):

        if self.frame_cnt % self.det_frequency == 0:
            bboxes, det_scores = self.det_model(image,score_thr=0.6)
        else:
            bboxes = self.bboxes_last_frame

        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        if not self.tracking:
            # without tracking

            bboxes_current_frame = []
            for kpts in keypoints:
                bbox = pose_to_bbox(kpts)
                bboxes_current_frame.append(bbox)
        else:
            # with tracking
            if len(self.track_ids_last_frame) == 0:
                self.next_id = len(self.bboxes_last_frame)
                self.track_ids_last_frame = list(range(self.next_id))

            bboxes_current_frame = []
            track_ids_current_frame = []
            for kpts in keypoints:
                bbox = pose_to_bbox(kpts)

                track_id, match_result = self.track_by_iou(bbox)

                if track_id > -1:
                    track_ids_current_frame.append(track_id)
                    bboxes_current_frame.append(bbox)

            self.track_ids_last_frame = track_ids_current_frame

        self.bboxes_last_frame = bboxes_current_frame
        self.frame_cnt += 1
        return keypoints, scores,track_ids_current_frame

    def track_by_iou(self, bbox):
        """Get track id using IoU tracking greedily.
        Args:
            bbox (list): The bbox info (left, top, right, bottom, score).
            next_id (int): The next track id.

        Returns:
            track_id (int): The track id.
            match_result (list): The matched bbox.
            next_id (int): The updated next track id.
        """

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        max_iou_score = -1
        max_index = -1
        match_result = None
        for index, each_bbox in enumerate(self.bboxes_last_frame):

            iou_score = compute_iou(bbox, each_bbox)
            print('iou',iou_score)
            if iou_score > max_iou_score:
                max_iou_score = iou_score
                max_index = index
        print(max_iou_score,area)
        if max_iou_score > self.tracking_thr:
            # if the bbox has a match and the IoU is larger than threshold
            track_id = self.track_ids_last_frame.pop(max_index)
            match_result = self.bboxes_last_frame.pop(max_index)
        elif area >= self.MIN_AREA:
            # no match, but the bbox is large enough,
            # assign a new track id
            track_id = self.next_id
            self.next_id += 1

        else:
            # if the bbox is too small, ignore it
            track_id = -1
        return track_id, match_result

def draw_skeleton(img,
                  keypoints,
                  scores,
                  openpose_skeleton=False,
                  kpt_thr=0.5,
                  radius=2,
                  line_width=2):
    num_keypoints = keypoints.shape[1]

    if openpose_skeleton:
        if num_keypoints == 18:
            skeleton = 'openpose18'
        elif num_keypoints == 134:
            skeleton = 'openpose134'
        elif num_keypoints == 26:
            skeleton = 'halpe26'
        else:
            raise NotImplementedError
    else:
        if num_keypoints == 17:
            skeleton = 'coco17'
        elif num_keypoints == 133:
            skeleton = 'coco133'
        elif num_keypoints == 21:
            skeleton = 'hand21'
        elif num_keypoints == 26:
            skeleton = 'halpe26'
        else:
            raise NotImplementedError

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    num_instance = keypoints.shape[0]
    if skeleton in ['coco17', 'coco133', 'hand21', 'halpe26']:
        for i in range(num_instance):
            img = draw_mmpose(img, keypoints[i], scores[i], keypoint_info,
                              skeleton_info, kpt_thr, radius, line_width)
    else:
        raise NotImplementedError
    return img
def draw_mmpose(img,
                keypoints,
                scores,
                keypoint_info,
                skeleton_info,
                kpt_thr=0.5,
                radius=2,
                line_width=2):
    assert len(keypoints.shape) == 2

    vis_kpt = [s >= kpt_thr for s in scores]

    link_dict = {}
    # print(keypoint_info)
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']
        kpt = keypoints[i]

        if vis_kpt[i]:
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius),
                             kpt_color, -1)
            img = cv2.putText(img,str(i),(int(kpt[0]), int(kpt[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5,kpt_color,1)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            link_color = ske_info['color']
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                           (int(kpt1[0]), int(kpt1[1])),
                           link_color,
                           thickness=line_width)

    return img

if __name__ == '__main__':
    import cv2

    pt = PoseTracker(det_path=r'C:\Users\84728\PycharmProjects\ai_sport_server\ai\models\rtmdet_nano.onnx',
                     pose_path=r'C:\Users\84728\PycharmProjects\ai_sport_server\ai\models\rtmpose_l.onnx')
    cap = cv2.VideoCapture(0)
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            break
        img_show = frame.copy()

        keypoints, scores,track_ids = pt(frame)

        img_show = draw_skeleton(img_show,
                                 keypoints,
                                 scores,
                                 kpt_thr=0.43)

        for i,track_id in enumerate(track_ids):
            img_show = cv2.putText(img_show, str(track_id),(int(keypoints[i][0][0]),int(keypoints[i][0][1])),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0,0),thickness=4)

        img_show = cv2.resize(img_show, (960, 540))
        cv2.imshow('img', img_show)
        cv2.waitKey(1)