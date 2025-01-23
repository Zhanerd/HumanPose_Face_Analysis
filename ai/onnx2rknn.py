from rknn.api import RKNN
import os
import cv2
import numpy
import onnxruntime
import numpy as np

arcface_dst = numpy.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=numpy.float32)


def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    tform, _ = cv2.estimateAffinePartial2D(lmk, dst)

    return tform
    # tform = trans.SimilarityTransform()
    # tform.estimate(lmk, dst)
    # M = tform.params[0:2, :]
    # return M


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def cosine_similarity(a, b):
    assert a.shape == b.shape

    # 计算点积
    dot_product = numpy.sum(a * b, axis=(-2, -1))

    # 计算范数
    norm_a = numpy.linalg.norm(a, axis=(-2, -1))
    norm_b = numpy.linalg.norm(b, axis=(-2, -1))

    # 计算余弦相似度
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def cosine_similarity2(a, b):
    assert a.shape == b.shape

    # 计算点积
    # dot_product = numpy.sum(a * b, axis=(-2, -1))

    # 计算范数
    norm_a = numpy.linalg.norm(a, axis=(-2, -1))
    norm_b = numpy.linalg.norm(b, axis=(-2, -1))
    #
    a = a / norm_a
    b = b / norm_b
    # 计算余弦相似度
    similarity = numpy.sum(a * b, axis=(-2, -1))
    return similarity


def euclidean_distance(a, b):
    return numpy.sqrt(numpy.sum((a - b) ** 2))


class convertRknn:
    def __init__(self, platform='rk3588', mean=None, std=None):
        self.model = RKNN()
        self.model.config(target_platform=platform, mean_values=mean, std_values=std)
        ### arcface mean_values=[[127.5, 127.5,127.5]], std_values=[[127.5, 127.5, 127.5]]

    def loading_onnx_model(self, model_path, input=None, input_size=None):
        print('--> Loading model')
        ret = self.model.load_onnx(model=model_path, inputs=input, input_size_list=input_size)
        if ret != 0:
            print('load model failed')
            exit(ret)
        print('loading_onnx_model done')

    def create_calibration_txt(self, calibration_path, calibration_txt):
        # 创建txt文件，并写入
        # calibration_path = '/home/hz/hz_lyb/onnx2rknn/calibration'
        # calibration_txt = 'calibration_dataset.txt'
        for root, subdir, files in os.walk(calibration_path):
            with open(os.path.join(calibration_path, calibration_txt), 'w') as f:
                for file in files:
                    f.write(f"{os.path.join(calibration_path, file)}\n")
                    # print(file)
        print('create_calibration_txt done')

    def building_model(self, need_quantization=False, dataset=None):
        print('-->Building model')
        if need_quantization:
            ret = self.model.build(do_quantization=need_quantization, dataset=dataset)  ### int8量化
        else:
            ret = self.model.build(do_quantization=need_quantization)
        if ret != 0:
            print('build model failed')
            exit()
        print('building_model done')
        self.init_runtime()

    def export_model(self, export_path):
        '''step 4: export and save the .rknn model'''
        print('--> Export RKNN model: {}'.format(export_path))
        ret = self.model.export_rknn(export_path)
        if ret != 0:
            print('Export rknn model failed.')
            exit(ret)
        print('done')

    def init_runtime(self):
        print('--> Init runtime environment')
        ret = self.model.init_runtime(target=None, device_id=None, perf_debug=True)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')

    def inference(self, img, fomat='nchw'):
        outputs = self.model.inference(inputs=img, data_format=fomat)[0]
        return outputs

    def release(self):
        print('release the model')
        self.model.release()
        print('done')


if __name__ == '__main__':
    rk = convertRknn()
    ### 默认3588
    rk.loading_onnx_model('yolov8n.onnx', input=['images'], input_size=[[1, 3, 320, 320]])
    rk.building_model()
    rk.export_model('yolov8n.rknn')

    # img = cv2.imread('/home/hz/hz_lyb/onnx2rknn/face_buf.jpg')
    # # 使用 blobFromImages 进行预处理
    # blob = cv2.dnn.blobFromImages(
    #     [img],
    #     scalefactor=1.0 / 127.5,
    #     size=(112, 112),
    #     mean=(127.5, 127.5, 127.5),
    #     swapRB=True
    # )
    # # 将 Blob 转换为 INT8（假设 scale=1.0, zero_point=-128）
    # img = numpy.expand_dims(img,axis=0)
    # img = img.transpose(0,3,1,2)
    # blob_int8 = [img]
    # # print(blob_int8)

    # rk = convertRknn(mean=[[127.5, 127.5,127.5]], std=[[127.5, 127.5, 127.5]])
    # rk.loading_onnx_model('/home/hz/hz_lyb/onnx2rknn/face_w600k_r50.onnx',['input.1'],[[1,3, 112,112]])
    # rk.building_model(True,'/home/hz/hz_lyb/onnx2rknn/calibration/calibration_dataset.txt')
    # # 使用accuracy_analysis 接口进行模型量化精度分析
    # rk.model.accuracy_analysis(
    #     inputs = ["face_buf.jpg"],               # inputs 表示进行推理的图像
    #     output_dir = 'snapshot',                          # 表示精度分析的输出目录
    #     target = None,                               # 表示目标硬件平台
    #     # target=None,                                      # 表示目标硬件平台
    #     device_id = None,                                 # 表示设备的编号
    # )

    # outputs = rk.inference(blob_int8)

    # session = onnxruntime.InferenceSession(path_or_bytes='face_w600k_r50.onnx',providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # sess_input = {session.get_inputs()[0].name: blob}
    # sess_output = []

    # for out in session.get_outputs():
    #     sess_output.append(out.name)

    # onnx_outputs = session.run(sess_output, sess_input)[0]

    # rk1 = convertRknn()
    # rk1.loading_onnx_model('/home/hz/hz_lyb/onnx2rknn/face_w600k_r50.onnx',['input.1'],[[1,3, 112,112]])
    # rk1.building_model(True,'/home/hz/hz_lyb/onnx2rknn/calibration/calibration_dataset.txt')
    # rk1.model.accuracy_analysis(
    #     inputs = ["face_buf.jpg"],               # inputs 表示进行推理的图像
    #     output_dir = 'snapshot2',                          # 表示精度分析的输出目录
    #     target = None,                               # 表示目标硬件平台
    #     # target=None,                                      # 表示目标硬件平台
    #     device_id = None,                                 # 表示设备的编号
    # )


