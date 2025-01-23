import time
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class PSPNet:
    def __init__(self, onnx_model):
        # 创建 Ort 推理会话，选择 CPU 或 GPU 提供者
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )
        # 根据 ONNX 模型类型选择 Numpy 数据类型（支持 FP32 和 FP16）
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

    # @staticmethod
    # def get_transform():
    #     transform_image_list = [
    #         transforms.Resize((256, 256), 3),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    #     return transforms.Compose(transform_image_list)

    def get_transform(self):
        def transform(image):
            # 调整图像大小
            shape = image.shape[:2]  # 原始图像大小
            new_shape = (self.model_height, self.model_width)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            ratio = r, r
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 填充宽高
            if shape[::-1] != new_unpad:  # 调整图像大小
                image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
            left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # 将图像从 BGR 格式转换为 RGB 格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 将图像转换为浮点数并归一化到 [0, 1] 范围
            image = image.astype(np.float32) / 255.0

            # 归一化处理
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std

            # 将图像从 HWC 格式转换为 CHW 格式
            image = image.transpose(2, 0, 1)

            image = image.astype(self.ndtype)

            return image

        return transform

    def preprocess(self, image):
        # 调整图像大小
        shape = image.shape[:2]  # 原始图像大小
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 填充宽高
        if shape[::-1] != new_unpad:  # 调整图像大小
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 将图像从 BGR 格式转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将图像转换为浮点数并归一化到 [0, 1] 范围
        image = image.astype(np.float32) / 255.0

        # 归一化处理
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # 将图像从 HWC 格式转换为 CHW 格式
        image = image.transpose(2, 0, 1)

        image = image.astype(self.ndtype)

        return image


    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        将掩膜缩放至原始图像大小
        Args:
            masks (np.ndarray): 缩放和填充后的掩膜
            im0_shape (tuple): 原始图像大小
            ratio_pad (tuple): 填充与原始图像的比例
        Returns:
            masks (np.ndarray): 缩放后的掩膜
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # 计算比例
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # 比例
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # 填充
        else:
            pad = ratio_pad[1]

        # 计算掩膜的边界
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" 应该是 2 或 3，但得到 {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # 使用 INTER_LINEAR 插值调整大小
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks


    def postprocess(self, preds,  im0_shape):
        # 缩放边界框，使其与原始图像尺寸匹配
        pred = preds[0]
        pred = pred.squeeze()
        pred = pred.transpose(1, 2, 0)

        # pred = np.where(pred < -1, 0, pred)

        masks = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))

        # masks = np.where(masks < 18, 0, masks)

        masks = self.scale_mask(masks, im0_shape)
        return masks

    def __call__(self, im0):
        # 图像预处理
        # im,ratio, (pad_w, pad_h) = self.preprocess(im0)
        im = self.preprocess(im0)
        im = np.expand_dims(im, axis=0)
        # ONNX 推理
        pred = self.session.run(None, {self.session.get_inputs()[0].name: im})
        pred = self.postprocess(pred, im0_shape=im0.shape)
        return pred

if __name__ == '__main__':
    net = PSPNet(onnx_model='PSPNet.onnx')
    img = cv2.imread(r'C:\Users\84728\Desktop\Single-Human-Parsing-LIP-master\demo\test.jpg')
    inference_time = time.time()
    pred = net(img)
    print('inference time: %f' % (time.time() - inference_time))
    # show_image(img, pred)
    cv2.imshow("frame", pred)
    cv2.waitKey(0)

