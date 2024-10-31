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

halp2hz = [
    0,   # NOSE
    2,   # LEFT_EYE
    1,   # RIGHT_EYE
    4,   # LEFT_EAR
    3,   # RIGHT_EAR
    6,   # LEFT_SHOULDER
    5,   # RIGHT_SHOULDER
    8,   # LEFT_ELBOW
    7,   # RIGHT_ELBOW
    10,  # LEFT_WRIST
    9,   # RIGHT_WRIST
    12,  # LEFT_HIP
    11,  # RIGHT_HIP
    14,  # LEFT_KNEE
    13,  # RIGHT_KNEE
    16,  # LEFT_ANKLE
    15,  # RIGHT_ANKLE
    25,  # LEFT_HEEL
    24,  # RIGHT_HEEL
    23,  # LEFT_FOOT_INDEX     ###目前取左脚外侧的点，如果要取左脚内点的话，需要修改为21，，如果要取中心点，要重新计算点在输入
    22   # RIGHT_FOOT_INDEX    ###目前取右脚外侧的点，如果要取右脚内点的话，需要修改为20，，如果要取中心点，要重新计算点在输入
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


def halp2hzlandmark(halp_kps):
    hz_kps = []
    if len(halp_kps) != 26:
        return hz_kps

    pose_2d = halp_kps[halp2hz]
    z_axis = np.zeros((pose_2d.shape[0], 1))
    pose_3d = np.hstack((pose_2d, z_axis))
    return pose_3d