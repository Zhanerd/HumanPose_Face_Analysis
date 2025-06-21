import os

import cv2
from ai.face_recognition.face_analysis import FaceAnalysis
import numpy as np
from ai.face_recognition.normal import normalize

### 和face_reco不同在于增加了人脸库对比
class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=0.6, det_thresh=0.5, det_size=(640, 640),det_path='./buffalo_l/det_10g.onnx',reg_path='./buffalo_l/w600k_r50.onnx'):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID，负数为使用CPU
        :param face_db: 人脸库文件夹
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size

        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        self.model = FaceAnalysis(det_onnx=det_path, reg_onnx=reg_path)
        self.model.prepare(gpu_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # 人脸库的人脸特征
        self.faces_embedding = list()
        # 加载人脸库中的人脸
        self.load_faces(self.face_db)

    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                user_name = file.split(".")[0]
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = normalize(embedding)
                self.faces_embedding.append({
                    "user_name": user_name,
                    "feature": embedding
                })

    # def estimate_norm(self,kps):
    #     arcface_dst = np.array(
    #         [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    #          [41.5493, 92.3655], [70.7299, 92.2041]],
    #         dtype=np.float32)
    #     kps = np.array(kps, dtype=np.float32)
    #     # arcface_dst = np.array(
    #     #     [[46.2946, 51.6963], [66.5318, 51.5014], [56.0252, 71.7366],
    #     #      [41.5493, 92.3655], [70.7299, 92.2041]],
    #     #     dtype=np.float32)
    #     # tform = trans.SimilarityTransform()
    #     # tform.estimate(kps, arcface_dst)
    #     # return tform.params[0:2, :]
    #     tform, _ = cv2.estimateAffinePartial2D(kps, arcface_dst)
    #     return tform
    # def norm_crop(self,img, kps):
    #     M = self.estimate_norm(kps)
    #     return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

    def match_feature(self, in_embedding_np, group_embedding_np, label_list, thre=0.75):
        cos_similarity = np.dot(in_embedding_np, group_embedding_np.T)
        # 使用 np.argmax 找出每一行的最大值位置
        max_indices = np.argmax(cos_similarity, axis=1)
        # 使用最大值的位置索引从原数组中提取对应的最大值
        max_values = cos_similarity[np.arange(cos_similarity.shape[0]), max_indices]

        match_infos = []
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

    # 人脸识别
    def recognition(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = normalize(embedding)
            user_name = "unknown"
            max = 0
            for com_face in self.faces_embedding:
                r, sim = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r and sim > max:
                    user_name = com_face["user_name"]
                    max = sim
                    # print("cos_sim", sim)
                    # print('user_name',user_name)
            results.append((user_name, max))
        return results

    # 人脸特征相似度比对
    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        dot_product = np.dot(feature1, feature2.T)
        norm_a = np.linalg.norm(feature1)
        norm_b = np.linalg.norm(feature2)

        # 防止除以零错误
        if norm_a == 0 or norm_b == 0:
            return False, None

        cos_sim = dot_product / (norm_a * norm_b)
        cos_sim = (cos_sim+1)/2
        if cos_sim > threshold:
            return True,cos_sim[0][0]
        else:
            return False,cos_sim[0][0]

    # 对比人脸,将符合条件的人脸入库并保存为png格式
    def register(self, image, user_name):
        faces = self.model.get(image)
        if len(faces) != 1:
            return '图片检测不到人脸'
        # 判断人脸是否存在
        embedding = np.array(faces[0].embedding).reshape((1, -1))
        embedding = normalize(embedding)
        is_exits = False
        for com_face in self.faces_embedding:
            r,_ = self.feature_compare(embedding, com_face["feature"], self.threshold)
            if r:
                is_exits = True
        if is_exits:
            return '该用户已存在'
        # 符合注册条件保存图片，同时把特征添加到人脸特征库中
        cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, '%s.png' % user_name))
        self.faces_embedding.append({
            "user_name": user_name,
            "feature": embedding
        })
        return "success"

    # 检测人脸
    def detect(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            result = dict()
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            #result["landmark_3d_68"] = np.array(face.landmark_3d_68).astype(np.int32).tolist()
            #result["landmark_2d_106"] = np.array(face.landmark_2d_106).astype(np.int32).tolist()
            #result["pose"] = np.array(face.pose).astype(np.int32).tolist()
            #result["age"] = face.age
            # gender = '男'
            # if face.gender == 0:
            #     gender = '女'
            # result["gender"] = gender
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = normalize(embedding)
            result["embedding"] = embedding
            results.append(result)
        return results

    @staticmethod
    def evaluate(detected_labels, actual_labels):
        detected_labels = set(detected_labels)
        actual_labels = set(actual_labels)


        # 计算 True Positives (TP)
        true_positives = actual_labels.intersection(detected_labels)
        TP = len(true_positives)

        # # 计算 True Negatives (TN)
        # true_negatives = actual_labels.difference(true_positives)
        # TN = len(true_negatives)

        # 计算 False Positives (FP)
        false_positives = detected_labels.difference(actual_labels)
        FP = len(false_positives)

        # 计算 False Negatives (FN)
        false_negatives = actual_labels.difference(detected_labels)
        FN = len(false_negatives)

        # 计算 Precision 和 Recall和F1-score
        # accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + FP + FN + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        lamda = 0.5
        f1_score = (recall * precision) / (lamda * recall + (1 - lamda) * precision) if (lamda * recall + (1 - lamda) * precision) > 0 else 0

        return precision, recall, f1_score




if __name__ == '__main__':
    # # 人脸注册
    # for subdir, _, files in os.walk(r'C:\Users\84728\Desktop\hz'):
    #     for file in files:
    #         # 检查文件扩展名
    #         if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    #             # 构造图像文件的完整路径
    #             image_path = os.path.join(subdir, file)
    #             # 一个图片对应一个文件夹
    #             image = cv2.imread(image_path)
    #             result = face_recognitio.register(image, user_name=file.split('.')[0])


    cap = cv2.VideoCapture(r'C:\Users\84728\Desktop\test\complicate.mp4')
    db = np.load(r'C:\Users\84728\Desktop\back\face\face_db.npy', allow_pickle=True)
    pid = [item.get('user_name') for item in db]
    P = list()
    ge = [item.get('feature') for item in db]
    ge_np = np.array(ge, dtype=float)
    ge_np = ge_np.reshape(78, 512)
    face_recognitio = FaceRecognition()

    img = cv2.imdecode(np.fromfile(r'C:\Users\84728\Desktop\test\run1.jpeg', dtype=np.uint8), -1)
    face_recognitio = FaceRecognition()
    # faces = face_recognitio.detect(img)
    # in_embed = [item.get('embedding') for item in faces]
    # in_embed = np.array(in_embed, dtype=float)
    # in_embed = np.squeeze(in_embed)
    #
    # print('recognizing')
    # print('in_embed', len(in_embed))
    # result = face_recognitio.match_feature(in_embed, ge_np, pid, 0.65)

    f = 1
    avg = 0
    id_list = []
    max_similarity = {}
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is False:
            break

        if frame is None:
            continue
        print('detecting')
        faces = face_recognitio.detect(frame)
        in_embed = [item.get('embedding') for item in faces]
        in_embed = np.array(in_embed, dtype=float)
        in_embed = np.squeeze(in_embed)
        if faces is None:
            continue

        print('recognizing')
        print('in_embed',len(in_embed))
        result = face_recognitio.match_feature(in_embed,ge_np,pid,0.65)
        Pred = [item.get('person_id') for item in result]

        diff_elements = set(Pred).difference(set(P))
        P.extend(diff_elements)



    gt = ['hw', 'lyb', 'ych']
    print(face_recognitio.evaluate(P, gt))
    print(P)