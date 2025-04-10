import onnx
import onnxruntime
import numpy as np
import cv2
import torch

class ArcFace:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        if 'onnx' in self.model_file:
            self.backend = 'onnxruntime'
            find_sub = False
            find_mul = False
            model = onnx.load(self.model_file)
            graph = model.graph
            for nid, node in enumerate(graph.node[:8]):
                # print(nid, node.name)
                if node.name.startswith('Sub') or node.name.startswith('_minus'):
                    find_sub = True
                if node.name.startswith('Mul') or node.name.startswith('_mul'):
                    find_mul = True
            if find_sub and find_mul:
                # mxnet arcface model
                input_mean = 0.0
                input_std = 1.0
            else:
                input_mean = 127.5
                input_std = 127.5
            self.input_mean = input_mean
            self.input_std = input_std
            # print('input mean and std:', self.input_mean, self.input_std)
            if self.session is None:
                self.session = onnxruntime.InferenceSession(self.model_file, None)
            input_cfg = self.session.get_inputs()[0]
            input_shape = input_cfg.shape
            input_name = input_cfg.name
            self.input_size = tuple(input_shape[2:4][::-1])
            self.input_shape = input_shape
            outputs = self.session.get_outputs()
            output_names = []
            for out in outputs:
                output_names.append(out.name)
            self.input_name = input_name
            self.output_names = output_names
            assert len(self.output_names) == 1
            self.output_shape = outputs[0].shape
        elif 'engine' in self.model_file:
            self.backend = 'tensorrt'
            input_mean = 127.5
            input_std = 127.5
            self.input_mean = input_mean
            self.input_std = input_std
            self.input_shape = ['None', 3, 112, 112]
            self.input_size = (112,112)
        else:
            print('unknown model type')


    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0 and self.backend == 'onnxruntime':
            self.session.set_providers(['CPUExecutionProvider'])

    # 处理一张人脸
    def get(self, img, face):
        aimgs = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        embed = self.get_feat(aimgs)
        if isinstance(embed, list):
            embed = embed[0]
        face.embedding = embed.flatten()
        return face.embedding

    # 处理多张脸
    def batch_get(self, img, faces):
        aimgs = list()
        for face in faces:
            aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
            aimgs.append(aimg)
        embeds = self.get_feat(aimgs)
        for i, embed in enumerate(embeds):
            faces[i].embedding = embed.flatten()
        return faces
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
        if self.backend == 'onnxruntime':
            net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        elif self.backend == 'tensorrt':
            blob = torch.from_numpy(blob).to('cuda')
            with torch.no_grad():
                outputs = self.session(blob)
            net_out = [output.cpu().numpy() for output in outputs]
        else:
            print('unknown backend')
            return
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        if self.backend == 'onnxruntime':
            net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        elif self.backend == 'tensorrt':
            blob = torch.from_numpy(blob).to('cuda')
            with torch.no_grad():
                outputs = self.session(blob)
            net_out = [output.cpu().numpy() for output in outputs][0]
        else:
            print('unknown backend')
            return
        return net_out

    def release(self):
        if self.backend == 'onnxruntime':
            del self.session
        elif self.backend == 'tensorrt':
            del self.session.engine
            del self.session.context
            import gc
            gc.collect()
        else:
            pass
        self.session = None


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
