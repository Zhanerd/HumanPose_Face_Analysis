import numpy as np

hzlandmark = {
    'NOSE': 0,
    'LEFT_EYE': 1,
    'RIGHT_EYE': 2,
    'LEFT_EAR': 3,
    'RIGHT_EAR': 4,
    'LEFT_SHOULDER': 5,
    'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7,
    'RIGHT_ELBOW': 8,
    'LEFT_WRIST': 9,
    'RIGHT_WRIST': 10,
    'LEFT_HIP': 11,
    'RIGHT_HIP': 12,
    'LEFT_KNEE': 13,
    'RIGHT_KNEE': 14,
    'LEFT_ANKLE': 15,
    'RIGHT_ANKLE': 16,
    'LEFT_HEEL': 17,
    'RIGHT_HEEL': 18,
    'LEFT_FOOT_INDEX': 19,
    'RIGHT_FOOT_INDEX': 20
}

halp_keypoints_reordered = [
    0,   # NOSE
    1,   # LEFT_EYE
    2,   # RIGHT_EYE
    3,   # LEFT_EAR
    4,   # RIGHT_EAR
    5,   # LEFT_SHOULDER
    6,   # RIGHT_SHOULDER
    7,   # LEFT_ELBOW
    8,   # RIGHT_ELBOW
    9,   # LEFT_WRIST
    10,  # RIGHT_WRIST
    11,  # LEFT_HIP
    12,  # RIGHT_HIP
    13,  # LEFT_KNEE
    14,  # RIGHT_KNEE
    15,  # LEFT_ANKLE
    16,  # RIGHT_ANKLE
    24,  # LEFT_HEEL
    25,  # RIGHT_HEEL
    22,  # LEFT_FOOT_INDEX (left_small_toe in HALP)
    23   # RIGHT_FOOT_INDEX (right_small_toe in HALP)
]

whole2hz = [
    0,   # NOSE
    1,   # LEFT_EYE
    2,   # RIGHT_EYE
    3,   # LEFT_EAR
    4,   # RIGHT_EAR
    5,   # LEFT_SHOULDER
    6,   # RIGHT_SHOULDER
    7,   # LEFT_ELBOW
    8,   # RIGHT_ELBOW
    9,  # LEFT_WRIST
    10,   # RIGHT_WRIST
    11,  # LEFT_HIP
    12,  # RIGHT_HIP
    13,  # LEFT_KNEE
    14,  # RIGHT_KNEE
    15,  # LEFT_ANKLE
    16,  # RIGHT_ANKLE
    19,  # LEFT_HEEL
    22,  # RIGHT_HEEL
    17,  # LEFT_FOOT_INDEX     ###目前取左脚外侧的点，如果要取左脚内点的话，需要修改为19，，如果要取中心点，要重新计算点在输入
    20   # RIGHT_FOOT_INDEX    ###目前取右脚外侧的点，如果要取右脚内点的话，需要修改为22，，如果要取中心点，要重新计算点在输入
]


halp = {
    'NOSE': 0,
    'LEFT_EYE': 1,
    'RIGHT_EYE': 2,
    'LEFT_EAR': 3,
    'RIGHT_EAR': 4,
    'LEFT_SHOULDER': 5,
    'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7,
    'RIGHT_ELBOW': 8,
    'LEFT_WRIST': 9,
    'RIGHT_WRIST': 10,
    'LEFT_HIP': 11,
    'RIGHT_HIP': 12,
    'LEFT_KNEE': 13,
    'RIGHT_KNEE': 14,
    'LEFT_ANKLE': 15,
    'RIGHT_ANKLE': 16,
    'HEAD': 17,
    'NECK': 18,
    'HIP': 19,
    'LEFT_BIG_TOE': 20,
    'RIGHT_BIG_TOE': 21,
    'LEFT_SMALL_TOE': 22,
    'RIGHT_SMALL_TOE': 23,
    'LEFT_HEEL': 24,
    'RIGHT_HEEL': 25
}

def halp2hz_reoredered_landmark(halp_kps):
    hz_kps = []
    if len(halp_kps) != 26:
        return hz_kps

    pose_2d = halp_kps[halp_keypoints_reordered]
    z_axis = np.zeros((pose_2d.shape[0], 1))
    pose_3d = np.hstack((pose_2d, z_axis))
    return pose_3d

def halp2hz_reoredered_scores(halp_scores):
    hz_scores = []
    if len(halp_scores) != 26:
        return hz_scores
    hz_scores = halp_scores[halp_keypoints_reordered]
    return hz_scores

def whole2hzlandmark(whole_kps):
    hz_kps = []
    if len(whole_kps) != 133:
        return hz_kps

    pose_2d = whole_kps[whole2hz]
    z_axis = np.zeros((pose_2d.shape[0], 1))
    pose_3d = np.hstack((pose_2d, z_axis))
    return pose_3d

def whole2hzscores(whole_scores):
    hz_scores = []
    if len(whole_scores) != 133:
        return hz_scores
    hz_scores = whole_scores[whole2hz]
    return hz_scores