import onnx
import onnxruntime
import numpy as np
import cv2

class ArcFaceONNX:
    def __init__(self, model_file=None, device_id=0):
        assert model_file is not None
        self.engine = sail.Engine(model_file, device_id, sail.IOMode.SYSO)

        self.handle = self.engine.get_handle()
        self.graph_name  = self.engine.get_graph_names()[0]

        # IO 名 / 形状 / dtype
        self.input_name  = self.engine.get_input_names (self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]  # 假设只 1 输出：N×15
        in_shape  = self.engine.get_input_shape (self.graph_name, self.input_name)
        in_dtype  = self.engine.get_input_dtype (self.graph_name, self.input_name)
        out_shape = self.engine.get_output_shape(self.graph_name, self.output_name)
        out_dtype = self.engine.get_output_dtype(self.graph_name, self.output_name)

        # 创建 Tensor（Host+Device）
        self.in_tensor  = sail.Tensor(self.handle, in_shape,  in_dtype, False, True)
        self.out_tensor = sail.Tensor(self.handle, out_shape, out_dtype, True,  True)

        self.taskname = 'recognition'

        input_mean = 127.5
        input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_shape = ['None', 3, 112, 112]
        self.input_size = (112,112)


    def get(self, img, face):
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        self.engine.process(
            self.graph_name,
            {self.input_name:  self.in_tensor},
            {self.output_name: self.out_tensor}:
        )
        
        net_out = self.out_tensor.asnumpy().squeeze(0)

        return net_out


arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x

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
