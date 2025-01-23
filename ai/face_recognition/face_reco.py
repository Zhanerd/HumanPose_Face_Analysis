import math
import time
import torch
import cv2
import numpy as np
import os
import sys

sys.path.append('/../../')
from ai.face_recognition.normal import normalize
from ai.face_recognition.face_analysis import FaceAnalysis
from ai.face_recognition.face_quality.face_quality import face_quality_assessment
from ai.face_recognition.face_direction.face_direct import FaceDirection

class FaceRecognition:
    def __init__(self, det_path="", reg_path="", gpu_id=0, det_thresh=0.5, det_size=(640, 640), quality_path=None, direct_path=None):
        """
        人脸识别工具类参数
        :param det_path: 人脸检测模型路径
        :param reg_path: 人脸特征提取模型路径
        :param gpu_id: <0为cpu,>=0为gpu
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        :param quality_path: 人脸质量评分模型
        :param direct_path: 人脸朝向模型
        """
        self.quality_model = None
        self.gpu_id = gpu_id
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.model = None

        # 是否进行人脸质量检测
        self.quality_path = quality_path
        self.direct_path = direct_path
        # 加载人脸识别模型
        self.is_init = False
        self.init(det_path=det_path, reg_path=reg_path, gpu_id=gpu_id)

    def init(self, det_path, reg_path, gpu_id=0):
        if os.path.exists(det_path) and os.path.exists(reg_path):
            self.gpu_id = gpu_id
            self.model = FaceAnalysis(det_path=det_path, reg_path=reg_path)
            self.model.prepare(gpu_id=gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
            if self.model is None:
                print(det_path)
                print(reg_path)
                print("face model init fail")
            else:
                #print("face model init success")
                self.is_init = True

        if self.quality_path is not None:
            self.quality_model = face_quality_assessment(quality_path=self.quality_path, gpu_id=gpu_id)

        if self.direct_path is not None:
            self.direct_path = FaceDirection(direct_path=self.direct_path, gpu_id=gpu_id)

        test = np.ones(shape=self.det_size, dtype=np.uint8)
        test = np.expand_dims(test, axis=-1)
        test = np.repeat(test, 3, axis=-1)
        self.detect(test, det_thresh=0.4, nms_thr=0.5)

    def init_state(self):
        return self.is_init

    # 检测人脸
    def detect(self, image, det_thresh=0.5, nms_thr=None, top_n=1, quality=False, quality_thre=0.5,std_thre=0.35):
        w,h = image.shape[:2]
        if w < self.det_size[0] or h < self.det_size[1]:
            padw = self.det_size[0] - w
            padh = self.det_size[0] - h
            img = cv2.copyMakeBorder(image, 0, max(padw,0), 0, max(padh,0), cv2.BORDER_CONSTANT, value=[0, 0, 0])
            faces = self.model.batch_get(img, det_thresh, nms_thr)
        else:
            faces = self.model.batch_get(image, det_thresh, nms_thr)
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
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            if self.quality_model and quality:
                result["bbox"] = [0 if i < 0 else i for i in result["bbox"]]
                face_img = image[result["bbox"][1]:result["bbox"][3], result["bbox"][0]:result["bbox"][2]]
                if self.direct_path is not None:
                    pitch, yaw, roll = self.direct_path.inference(face_img)
                else:
                    pitch, yaw, roll = self.face_direction(result, image.shape[:2])
                scores = self.quality_model.inference(face_img)
                if not self.is_centered(result["bbox"], width, height):
                    result["error_code"] = 100  # 人脸未在中间
                    result["error_message"] = "人脸未在中间"
                elif not self.bbox_ratio(result["bbox"], width, height):
                    result["error_code"] = 200  # 人脸框过小
                    result["error_message"] = "人脸框过小"
                elif 75 < abs(pitch) < 80:
                    # print("pitch",pitch)
                    result["error_code"] = 301  # 仰角过大
                    result["error_message"] = "仰角过大 {}".format(abs(pitch))
                elif 150 > abs(yaw) > 80:
                    # print("yaw", yaw)
                    result["error_code"] = 302  # 偏航角过大
                    result["error_message"] = "偏航角过大 {}".format(abs(yaw))
                elif 65 < abs(roll) < 70:
                    # print("roll", roll)
                    result["error_code"] = 303  # 翻滚角过大
                    result["error_message"] = "翻滚角过大 {}".format(abs(roll))
                else:
                    score = round(np.mean(scores), 3)
                    std = np.std(scores)
                    result["scores"] = score
                    if score < quality_thre or std > std_thre:  # 加入方差确保质量的稳定
                        # print("face_scores", std, score)
                        result["error_code"] = 400  # 人脸置信度过低，可解释为有遮挡或者光线不良或者清晰度不佳
                        result["error_message"] = "有遮挡或者光线不良或者清晰度不佳,  score : {}, std: {}".format(score, std)
                    else:
                        result["error_code"] = 0
                        result["error_message"] = "success"
            result["det_score"] = round(float(face.det_score), 2)
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = normalize(embedding)
            result["embedding"] = embedding
            results.append(result)
        return results

    def detect_queue(self, image, det_thresh=0.5, nms_thr=None, top_n=1, quality=False, quality_thre=0.5,std_thre=0.35):
        w,h = image.shape[:2]
        if w < self.det_size[0] or h < self.det_size[1]:
            padw = self.det_size[0] - w
            padh = self.det_size[0] - h
            img = cv2.copyMakeBorder(image, 0, max(padw,0), 0, max(padh,0), cv2.BORDER_CONSTANT, value=[0, 0, 0])
            faces = self.model.batch_get_queue(img, det_thresh, nms_thr)
        else:
            faces = self.model.batch_get_queue(image, det_thresh, nms_thr)
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
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            if self.quality_model and quality:
                result["bbox"] = [0 if i < 0 else i for i in result["bbox"]]
                face_img = image[result["bbox"][1]:result["bbox"][3], result["bbox"][0]:result["bbox"][2]]
                if self.direct_path is not None:
                    pitch, yaw, roll = self.direct_path.inference(face_img)
                else:
                    pitch, yaw, roll = self.face_direction(result, image.shape[:2])
                scores = self.quality_model.inference(face_img)
                if not self.is_centered(result["bbox"], width, height):
                    result["error_code"] = 100  # 人脸未在中间
                    result["error_message"] = "人脸未在中间"
                elif not self.bbox_ratio(result["bbox"], width, height):
                    result["error_code"] = 200  # 人脸框过小
                    result["error_message"] = "人脸框过小"
                elif 75 < abs(pitch) < 80:
                    # print("pitch",pitch)
                    result["error_code"] = 301  # 仰角过大
                    result["error_message"] = "仰角过大 {}".format(abs(pitch))
                elif 150 > abs(yaw) > 80:
                    # print("yaw", yaw)
                    result["error_code"] = 302  # 偏航角过大
                    result["error_message"] = "偏航角过大 {}".format(abs(yaw))
                elif 65 < abs(roll) < 70:
                    # print("roll", roll)
                    result["error_code"] = 303  # 翻滚角过大
                    result["error_message"] = "翻滚角过大 {}".format(abs(roll))
                else:
                    score = round(np.mean(scores), 3)
                    std = np.std(scores)
                    result["scores"] = score
                    if score < quality_thre or std > std_thre:  # 加入方差确保质量的稳定
                        # print("face_scores", std, score)
                        result["error_code"] = 400  # 人脸置信度过低，可解释为有遮挡或者光线不良或者清晰度不佳
                        result["error_message"] = "有遮挡或者光线不良或者清晰度不佳,  score : {}, std: {}".format(score, std)
                    else:
                        result["error_code"] = 0
                        result["error_message"] = "success"
            result["det_score"] = round(float(face.det_score), 2)
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = normalize(embedding)
            result["embedding"] = embedding
            results.append(result)
        return results

    # 比对特征
    def match_feature(self, in_embedding_np, group_embedding_np, label_list, thre=0.75):
        match_infos = []
        if not isinstance(in_embedding_np, np.ndarray) or not isinstance(group_embedding_np, np.ndarray):
            print("in_embedding_np or group_embedding_np is not np.ndarray")
            return match_infos
        if label_list == [] or group_embedding_np.size == 0 or in_embedding_np.size == 0:
            return match_infos

        cos_similarity = np.dot(in_embedding_np, group_embedding_np.T)
        # 使用 np.argmax 找出每一行的最大值位置
        max_indices = np.argmax(cos_similarity, axis=1)
        # 使用最大值的位置索引从原数组中提取对应的最大值
        max_values = cos_similarity[np.arange(cos_similarity.shape[0]), max_indices]

        for i in range(0, max_values.shape[0], 1):
            match_info = dict()
            similarity = (max_values[i] + 1) / 2
            # print("similarity : ")
            # print(similarity)
            index = max_indices[i]
            if similarity < thre:
                match_info["person_id"] = ""
                match_info["similarity"] = 0
            else:
                match_info["person_id"] = label_list[index]
                match_info["similarity"] = round(similarity, 3)
            match_infos.append(match_info)
        return match_infos

    def match_feature_tensor_optimized(self, group_id, embedding, thre=0.75):
        # if not label_list or group_embedding.size == 0 or in_embedding.size == 0:
        #     return []
        # # 将张量移动到 GPU 上，使用 float32，并确保内存连续
        # ### 如果提前转移到gpu会更快
        # in_embedding = torch.from_numpy(in_embedding).to('cuda').float().contiguous()
        # group_embedding = torch.from_numpy(group_embedding).to('cuda').float().contiguous()
        match_infos = []
        dict_features = self.query_group_features(group_id)
        if dict_features is None:
            return match_infos
        group_embedding = dict_features["features"]
        if type(group_embedding) == np.ndarray:
            group_embedding = torch.from_numpy(group_embedding).to('cuda').float().contiguous()
        if type(embedding) == np.ndarray:
            embedding = torch.from_numpy(embedding).to('cuda').float().contiguous()
        label_list = dict_features["pids"]
        faceid_list = dict_features["fids"]
        with torch.no_grad():
            # 计算余弦相似度
            cos_similarity = torch.matmul(embedding, group_embedding.T)
            # 找出每一行的最大值及其索引
            max_values, max_indices = torch.max(cos_similarity, dim=1)

            for i in range(0, max_values.shape[0], 1):
                match_info = dict()
                similarity = (max_values[i] + 1) / 2
                # print("similarity : ")
                # print(similarity)
                index = max_indices[i]
                if similarity < thre:
                    match_info["person_id"] = ""
                    match_info["face_id"] = ""
                    # match_info["similarity"] = 0
                else:
                    match_info["person_id"] = label_list[index]
                    match_info["face_id"] = faceid_list[index]
                match_info["similarity"] = round(similarity.cpu().tolist(), 3)
                match_infos.append(match_info)
        return match_infos


    def is_centered(self, bbox, img_width, img_height, threshold_percentage=0.15):
        x1, y1, x2, y2 = bbox
        bbox_center_x = x1 + (x2 - x1) / 2
        bbox_center_y = y1 + (y2 - y1) / 2
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
        bbox_area = (x2 - x1) * (y2 - y1)
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
            (0.0, 0.0, 0.0),  # Nose tip
            (-225.0, 170.0, -135.0),  # Left eye
            (225.0, 170.0, -135.0),  # Right eye
            (0.0, 0.0, 0.0),  # Nose tip
            (-150.0, -150.0, -125.0),  # Left Mouth
            (150.0, -150.0, -125.0),  # Right mouth
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
        return Y, X, Z

    def release(self):
        self.model.release()


if __name__ == '__main__':
    face_recognitio = FaceRecognition(det_path=r'D:\ai\ai\models\face_det_10g.engine',
                                      reg_path=r'D:\ai\ai\models\face_w600k_r50.engine',
                                      quality_path = r"D:\ai\ai\models\face_quality_assessment.onnx",
                                      gpu_id=0)
    # #img = cv2.imdecode(np.fromfile('http://192.168.7.170:8077/file/20240829/814a4790ead94cfaae34b5f5029f6d4c.jpg', dtype=np.uint8), -1)
    # img = cv2.imread(r'D:\ai\ai\models\bcaa68ec0c3146f4b5b32c606ab0909a.png')
    # t1 = time.time()
    # results = face_recognitio.detect(img, det_thresh=0.4, nms_thr=0.5, quality=True)
    # print('face with quailty time ', time.time() - t1)
    # print(results)

    # cap = cv2.VideoCapture(0)
    # a = time.time()
    # count = 0
    # while count<10:
    #     ret, frame = cap.read()
    #     results = face_recognitio.detect_queue(frame, det_thresh=0.4, nms_thr=0.5, quality=True)
    #     for result in results:
    #         frame = cv2.rectangle(frame, (result['bbox'][0], result['bbox'][1]), (result['bbox'][2], result['bbox'][3]),
    #                               (0, 0, 255), 2)
    #     count += 1
    #     cv2.imshow('a',frame)
    #     cv2.waitKey(1)
    # print('queue',time.time()-a)

    frame = cv2.imread(r"C:\Users\84728\Desktop\test\t1.jpeg")
    def normal_detect(frame):
        count = 0
        a = time.time()
        while count<200:
            # ret, frame = cap.read()
            results = face_recognitio.detect(frame, det_thresh=0.4, nms_thr=0.5, quality=True)
            count += 1

        print('normal',time.time()-a)
    def queue_detect(frame):
        a = time.time()
        count = 0
        while count<200:
            results = face_recognitio.detect_queue(frame, det_thresh=0.4, nms_thr=0.5, quality=True)
            count += 1
        face_recognitio.model.faces_queue = []
        print('queue',time.time()-a)

    for i in range(10):
        normal_detect(frame)
        queue_detect(frame)
    # face_quality = face_quality_assessment('ai/models/face-quality-assessment.onnx', gpu_id=-1)
    # from utils.FrameCapture import FrameCapture
    # cap = FrameCapture(frame_source="rtsp://admin:hz123456@192.168.20.102:554/Streaming/Channels/1")
    # cap.connectVideoCapture()
    # while True:
    #     frame, ft = cap.getFrame()
    #     if frame is None:
    #         print(None)
    #         continue
    #     frame = cv2.resize(frame, (640, 480))
    #     results = face_recognitio.detect(frame,quality=True)
    #     for result in results:
    #         frame = cv2.rectangle(frame, (result['bbox'][0], result['bbox'][1]), (result['bbox'][2], result['bbox'][3]),
    #                               (0, 0, 255), 2)
    #         image = frame[result['bbox'][1]:result['bbox'][3], result['bbox'][0]:result['bbox'][2]]
    #         #scores = face_quality.inference(image)
    #         #score = round(np.mean(scores), 2)
    #         #frame = cv2.putText(frame, str(score), (result['bbox'][0], result['bbox'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    #         #print(scores, np.std(scores))
    #         cv2.imshow('image', frame)
    #         cv2.waitKey(1)

    # for result in results:
    #     print('人脸框坐标：{}'.format(result["bbox"]))
    #     print('人脸五个关键点：{}'.format(result["kps"]))
    #     print(result["errorcode"])

    # c50 = 0
    # c55 = 0
    # c60 = 0
    # c65 = 0
    # c70 = 0
    # c75 = 0
    # for current_dir,sub_dirs,files in os.walk(r'C:\Users\84728\Desktop\back\face\face_db'):
    #     for file in files:
    #         # 检查文件是否为图片
    #         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    #             # 构建完整的文件路径
    #             image_path = os.path.join(current_dir, file)
    #
    #             # 读取图片
    #             img = cv2.imread(image_path)
    #             # 人脸识别
    #             # print("检测ing")
    #             # faces_img = list()
    #             results = face_recognitio.detect(img,det_thresh=0.7,nms_thr=0.7,quality=True,top_n=1)
    #             for result in results:
    #                 if result["scores"]>0.5:
    #                     c50+=1
    #                     if result["scores"]>0.55:
    #                         c55+=1
    #                         if result["scores"]>0.6:
    #                             c60+=1
    #                             if result["scores"]>0.65:
    #                                 c65+=1
    #                                 if result["scores"]>0.7:
    #                                     c70+=1
    #                                     if result["scores"]>0.75:
    #                                         c75+=1
    #                 print('name',file)
    #                 print('人脸框坐标：{}'.format(result["bbox"]))
    #                 print('人脸五个关键点：{}'.format(result["kps"]))
    #                 print(result["errorcode"])
    # print(c50,c55,c60,c65,c70,c75)
    # print('e')
