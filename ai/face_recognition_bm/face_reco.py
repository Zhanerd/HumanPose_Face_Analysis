import math
import time

import cv2
import numpy as np
import os
from ai.face_recognition.normal import normalize
from ai.face_recognition.face_analysis import FaceAnalysis
from ai.face_recognition.face_quality import face_quality_assessment


class FaceRecognition:
    def __init__(self, det_path="", reg_path="", gpu_id=0, det_thresh=0.5, det_size=(640, 640), quality_path=None):
        """
        人脸识别工具类参数
        :param det_path: 人脸检测模型路径
        :param reg__path: 人脸特征提取模型路径
        :param device: cuda,cpu
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.quality_model = None
        self.gpu_id = gpu_id
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.model = None

        # 是否进行人脸质量检测
        self.quality_path = quality_path

        # 加载人脸识别模型
        self.is_init = False
        self.init(det_path=det_path, reg_path=reg_path, gpu_id=gpu_id)

    def init(self, det_path, reg_path, gpu_id=0):
        if os.path.exists(det_path) and os.path.exists(reg_path):
            self.gpu_id = gpu_id
            self.model = FaceAnalysis(det_bmodel=det_path, reg_bmodel=reg_path)
            self.model.prepare(det_thresh=self.det_thresh, input_size=self.det_size)
            if self.model is None:
                print(det_path)
                print(reg_path)
                print("face model init fail")
            else:
                #print("face model init success")
                self.is_init = True

        if self.quality_path is not None:
            self.quality_model = face_quality_assessment(quality_path=self.quality_path, gpu_id=gpu_id)

    def init_state(self):
        return self.is_init

    # 检测人脸
    def detect(self, image, det_thresh=0.5, nms_thr=None, top_n=0, quality=False,
               quality_thre=0.5, std_thre=0.4, need_feature=True):
        # w,h = image.shape[:2]
        # if w < self.det_size[0] and h < self.det_size[1]:
        # img = cv2.copyMakeBorder(image, 0, 528, 0, 368, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        faces = self.model.get(image, det_thresh, nms_thr)
        # 筛选最大的top_n个人脸
        if top_n > 0 and top_n < len(faces):
            def face_size(face):
                size = (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1])
                return -1 * size  # 从大到小排序

            faces.sort(key=face_size)
            faces = faces[:top_n]
        height, width = image.shape[:2]
        results = list()
        for face in faces:
            result = dict()
            result["error_code"] = 0
            result["error_message"] = "success"
            result["scores"] = 0.0
            result["std"] = 0.0
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            if self.quality_model and quality:
                result["bbox"] = [0 if i < 0 else i for i in result["bbox"]]
                face_img = image[result["bbox"][1]:result["bbox"][3], result["bbox"][0]:result["bbox"][2]]
                pitch, yaw, roll = self.face_direction(result, image.shape[:2])
                scores = self.quality_model.inference(face_img)
                if not self.is_centered(result["bbox"], width, height):
                    result["error_code"] = 100       # 人脸未在中间
                    result["error_message"] = "人脸未在中间"
                elif not self.bbox_ratio(result["bbox"], width, height):
                    result["error_code"] = 200       # 人脸框过小
                    result["error_message"] = "人脸框过小"
                elif 75 < abs(pitch) < 80:
                    # print("pitch",pitch)
                    result["error_code"] = 301     # 仰角过大
                    result["error_message"] = "仰角过大, {}".format(abs(pitch))
                elif 150 > abs(yaw) > 80:
                    # print("yaw", yaw)
                    result["error_code"] = 302     # 偏航角过大
                    result["error_message"] = "偏航角过大, {}".format(abs(yaw))
                elif 65 < abs(roll) < 70:
                    # print("roll", roll)
                    result["error_code"] = 303     # 翻滚角过大
                    result["error_message"] = "翻滚角过大, {}".format(abs(roll))
                else:
                    score = round(float(np.mean(scores)), 3)
                    std = float(np.std(scores))
                    result["scores"] = score
                    result["std"] = round(std, 3)
                    if score < quality_thre or std > std_thre:  # 加入方差确保质量的稳定
                        print("face_scores", std, score)
                        result["error_code"] = 400       # 人脸置信度过低，可解释为有遮挡或者光线不良或者清晰度不佳
                        result["error_message"] = "有遮挡或者光线不良或者清晰度不佳, score : {}, std: {}".format(score, std)
                    else:
                        result["error_code"] = 0
                        result["error_message"] = "success"
            result["det_score"] = round(float(face.det_score), 2)
            if need_feature:
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = normalize(embedding)
                result["embedding"] = embedding
            results.append(result)
        return results

    # 比对特征
    def match_feature(self, in_embedding_np, group_embedding_np, label_list, thre=0.75):
        match_infos = []
        match_index = -1
        if label_list == [] or group_embedding_np.size == 0 or in_embedding_np.size == 0:
            return match_infos, match_index

        cos_similarity = np.dot(in_embedding_np, group_embedding_np.T)
        # 使用 np.argmax 找出每一行的最大值位置
        max_indices = np.argmax(cos_similarity, axis=1)
        # 使用最大值的位置索引从原数组中提取对应的最大值
        max_values = cos_similarity[np.arange(cos_similarity.shape[0]), max_indices]
        #print("max_indices : {}".format(max_indices))
        #print("max_values : {}".format(max_values))
        for i in range(0, max_values.shape[0], 1):
            match_info = dict()
            match_info["face_id"] = ""
            similarity = (max_values[i] + 1) / 2
            print("similarity : ")
            print(similarity)
            index = max_indices[i]
            if similarity < thre:
                match_info["person_id"] = ""
                #match_info["similarity"] = 0
            else:
                match_info["person_id"] = label_list[index]
                match_index = index
                #match_info["similarity"] = round(similarity, 3)
            match_info["similarity"] = round(similarity, 3)
            match_infos.append(match_info)
        return match_infos, match_index

    def match_multi_feature(self, in_emb, db_emb, labels,
                            thre=0.75, top_k=-1):
        """
            返回满足阈值的匹配信息，并按相似度从高到低排序。
            - 若 top_k>0，仅保留相似度最高的 K 条。
            """
        # --- 参数检查 -----------------------------------------------------------
        if not labels or db_emb.size == 0 or in_emb.size == 0:
            return []

        # --- 计算 Cosine 相似度并归一到 [0,1] -----------------------------------
        #   假设输入已经 L2 归一化，否则请先 `in_emb /= np.linalg.norm(in_emb, axis=1, keepdims=True)` 处理
        sim = in_emb @ db_emb.T  # [B,N]
        sim = (sim + 1.0) * 0.5  # 映射到 [0,1]

        # --- 过滤阈值 -----------------------------------------------------------
        rows, cols = np.where(sim >= thre)
        if rows.size == 0:  # 没有符合阈值的
            return []

        sims = sim[rows, cols]  # 1-D 相似度数组
        if top_k > 0 and sims.size > top_k:
            # argpartition 只做 K 次比较，O(N)
            keep_idx = np.argpartition(-sims, top_k - 1)[:top_k]
            rows, cols, sims = rows[keep_idx], cols[keep_idx], sims[keep_idx]

        # --- 按相似度降序 -------------------------------------------------------
        order = np.argsort(-sims)  # 完全排序只在 K 条数据上做，开销极小
        rows, cols, sims = rows[order], cols[order], sims[order]

        labels_arr = np.asarray(labels)  # 切一次列表 → ndarray 便于索引
        match_infos = [
            {
                "person_id": labels_arr[c],
                "similarity": round(float(s), 3)
            }
            for c, s in zip(cols, sims)
        ]
        return match_infos

    # 返回多个特征
    def match_multi_old_feature(self, in_embedding_np, group_embedding_np, label_list, thre=0.75,top_k=-1):
        match_infos = []
        if not label_list or group_embedding_np.size == 0 or in_embedding_np.size == 0:
            return match_infos, -1

        # 计算余弦相似度并归一化到 [0, 1]
        cos_similarity = np.dot(in_embedding_np, group_embedding_np.T)
        similarity_matrix = (cos_similarity + 1) / 2

        for i in range(similarity_matrix.shape[0]):  # 对每个输入特征
            for j in range(similarity_matrix.shape[1]):  # 对每个库中的人脸特征
                similarity = similarity_matrix[i][j]
                if similarity >= thre:
                    match_info = {
                        "face_id": "",
                        "person_id": label_list[j],
                        "similarity": round(float(similarity), 3)
                    }
                    match_infos.append(match_info)

        # 按 similarity 从大到小排序
        match_infos.sort(key=lambda x: x["similarity"], reverse=True)
        if top_k>0:
            match_infos = match_infos[:top_k]
        return match_infos
    
    def is_centered(self, bbox, img_width, img_height, threshold_percentage=0.15):
        x1, y1, x2, y2 = bbox
        bbox_center_x = x1 + (x2-x1) / 2
        bbox_center_y = y1 + (y2-y1) / 2
        img_center_x = img_width / 2
        img_center_y = img_height / 2

        # 根据图像大小调整阈值
        threshold_x = img_width * threshold_percentage
        threshold_y = img_height * threshold_percentage

        # 判断检测框中心是否在图片中心附近
        if abs(bbox_center_x - img_center_x) <= threshold_x and abs(bbox_center_y - img_center_y) <= threshold_y:
            return True
        else:
            return False

    def bbox_ratio(self, bbox, img_width, img_height, threshold=0.05):
        x1, y1, x2, y2 = bbox
        bbox_area = (x2-x1) * (y2-y1)
        img_area = img_width * img_height
        ratio = bbox_area / img_area

        # 判断人脸框的大小
        if ratio >= threshold:
            return True
        else:
            return False

    def face_direction(self, result, img_size):
        image_points = np.array([
            (result['kps'][2][0], result['kps'][2][1]),  # Nose tip
            (result['kps'][0][0], result['kps'][0][1]),  # Left eye
            (result['kps'][1][0], result['kps'][1][1]),  # Right eye
            (result['kps'][2][0], result['kps'][2][1]),  # Nose tip
            (result['kps'][3][0], result['kps'][3][1]),  # Left Mouth
            (result['kps'][4][0], result['kps'][4][1]),  # Right mouth
        ], dtype=np.float32)
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose tip
            (-225.0, 170.0, -135.0),    # Left eye
            (225.0, 170.0, -135.0),     # Right eye
            (0.0, 0.0, 0.0),            # Nose tip
            (-150.0, -150.0, -125.0),   # Left Mouth
            (150.0, -150.0, -125.0),    # Right mouth
        ], dtype=np.float32)
        # Camera internals
        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        # calculate rotation angles
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)

        # transformed to quaterniond
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta

        ysqr = y * y
        # pitch (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        pitch = math.atan2(t0, t1)

        # yaw (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        if t2 > 1.0:
            t2 = 1.0
        if t2 < -1.0:
            t2 = -1.0
        yaw = math.asin(t2)

        # roll (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)

        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        # 单位转换：将弧度转换为度
        Y = int((pitch / math.pi) * 180)
        X = int((yaw / math.pi) * 180)
        Z = int((roll / math.pi) * 180)
        # euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(Y, X, Z)
        # print(euler_angle_str)

        return Y,X,Z
    def release(self):
        self.model.release()


if __name__ == '__main__':
    face_model = FaceRecognition(det_path='/home/admin/ydai/ai/models/face_det_10g.bmodel',
                    reg_path='/home/admin/ydai/ai/models/face_w600k_r50.bmodel',
                    quality_path='/home/admin/ydai/ai/models/face-quality-assessment.bmodel',
                    gpu_id=0)

    img = cv2.imread('/home/admin/ydai/ai/face_recognition/test_face/053.jpg')

    results = face_model.detect(img,quality=True)

    print(results)
