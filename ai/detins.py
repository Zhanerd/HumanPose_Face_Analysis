import time

import cv2
import numpy as np
import onnxruntime as ort


class Detins:
    def __init__(self, modelfile):
        self.mean = np.array([103.53, 116.28, 123.675])
        self.std = np.array([57.375, 57.12, 58.395])
        self.ort_session = None
        self.input_name = ""
        self.ort_session = ort.InferenceSession(
            modelfile,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )
        self.ndtype = np.half if self.ort_session.get_inputs()[0].type == "tensor(float16)" else np.single
        # 获取模型输入的名称
        self.input_name = self.ort_session.get_inputs()[0].name
        self.model_input_size = (640, 640)

    def preprocess(self, img):
        """
        图像预处理
        Args:
            img (Numpy.ndarray): 输入图像
        Returns:
            img_process (Numpy.ndarray): 处理后的图像
            ratio (tuple): 宽高比例
            pad_w (float): 宽度的填充
            pad_h (float): 高度的填充
        """
        # 调整输入图像大小并使用 letterbox 填充
        shape = img.shape[:2]  # 原始图像大小
        new_shape = (self.model_input_size[0], self.model_input_size[1])
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 填充宽高
        if shape[::-1] != new_unpad:  # 调整图像大小
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转换：HWC -> CHW -> BGR 转 RGB -> 除以 255 -> contiguous -> 添加维度
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process

    def __call__(self, image):
        shape = image.shape
        img = self.preprocess(image)
        out = self.inference(img)
        outputs = self.postprocess(out, shape)
        return outputs

    def inference(self, img):
        out = self.ort_session.run(None, {self.input_name: img})
        return out

    def postprocess(self, out, shape):
        out = out[2][0][0]
        out = (out > 0.5).astype(np.uint8) * 255
        out = self.scale_mask(out, shape)
        return out

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
            raise ValueError(f'"len of masks shape" 应该是 2 或 3 但得到 {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks,
            (im0_shape[1], im0_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks


if __name__ == '__main__':
    detins = Detins("detins.onnx")
    image_path = "land_in.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # 目标颜色和替换颜色
    # target_color = np.array([206, 183, 177])
    # replacement_color = np.array([0, 255, 0])
    #
    # # 颜色的容差范围 (可以根据需要调整)
    # tolerance = 2
    #
    # # 创建一个掩码，识别接近目标颜色的像素
    # mask = cv2.inRange(image, target_color - tolerance, target_color + tolerance)
    #
    # # 将符合条件的区域替换为绿色
    # image[mask > 0] = replacement_color
    #
    # cv2.imshow('show', image)
    # cv2.waitKey(0)
    # 获取image 宽高
    h, w, _ = image.shape
    # # 标准化 输出宽640
    # markWidth = 640
    # # 计算缩放比例
    # scale = markWidth / w
    # changeHeight = int(h * scale)
    # # 缩放图像
    # image = cv2.resize(image, (markWidth, changeHeight))
    t1 = time.time()
    output = detins(image)
    print(time.time() - t1)
    # 输出结果
    # array = output[2][0][0]

    # print('array',output.shape,np.max(output))

    # binary_image = image.copy()
    # binary_image = cv2.resize(binary_image, (640, 640))
    # # binary_image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    # binary_image[output == 0] = [239, 0, 0]  # RGB 黑色
    # # mask = array >= 0.5
    # # binary_image[mask] = (binary_image[mask] * (array[mask]-0.5)[:, np.newaxis]).astype(np.uint8)

    # binary_image = cv2.resize(binary_image, (w, h))
    # # # 创建一个两倍image宽的全白图像
    # # white_image = np.ones((changeHeight, markWidth * 2, 3), dtype=np.uint8) * 0
    # # white_image[:, :markWidth] = binary_image
    # # white_image[:, markWidth:markWidth * 2] = image
    # # 将binary_image保存本地图片

    # cv2.imwrite("binary_image.jpg", cv2.cvtColor(binary_image, cv2.COLOR_RGB2BGR))

    # # 显示图像
    # cv2.imshow('show', cv2.cvtColor(binary_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()