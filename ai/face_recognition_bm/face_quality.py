import cv2
import numpy as np
import sophon.sail as sail

class FaceQualityAssessment:
    def __init__(self, bmodel_path: str, device_id: int = 0):
        # ---------- 1. 基本参数 ----------
        self.device_id     = device_id
        self.in_h, self.in_w = 112, 112

        # ---------- 2. 加载 bmodel ----------
        # IoMode.SYSO 表示 Host 输入 / Host 输出
        self.engine = sail.Engine(bmodel_path, device_id, sail.IOMode.SYSO)

        # 取第一张 Graph
        self.graph_name  = self.engine.get_graph_names()[0]

        # ---------- 3. 拿 IO 名、shape、dtype ----------
        self.input_name  = self.engine.get_input_names (self.graph_name)[0]
        self.output_name = self.engine.get_output_names(self.graph_name)[0]

        in_shape  = self.engine.get_input_shape (self.graph_name, self.input_name)   # e.g. (1,3,112,112)
        in_dtype  = self.engine.get_input_dtype (self.graph_name, self.input_name)   # sail.DType.F32
        out_shape = self.engine.get_output_shape(self.graph_name, self.output_name)
        out_dtype = self.engine.get_output_dtype(self.graph_name, self.output_name)

        # ---------- 4. 句柄 & Tensor 预分配 ----------
        handle = self.engine.get_handle()
        self.in_tensor  = sail.Tensor(handle, in_shape,  in_dtype,  False, True)  # Device=False 表示在 Host
        self.out_tensor = sail.Tensor(handle, out_shape, out_dtype, True,  True )  # Device=True  自动 malloc device buf

    # -------------------------------------------------------------
    def _preprocess(self, img_bgr):
        """BGR → RGB → [0,1] → NCHW(float32)"""
        img = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                         (self.in_w, self.in_h))
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5   # [-1,1] 归一化
        img = np.transpose(img, (2, 0, 1))[None, ...]        # (1,3,112,112)
        return img

    # -------------------------------------------------------------
    def inference(self, src_bgr: np.ndarray) -> np.ndarray:
        # 1) 预处理
        host_in = self._preprocess(src_bgr)

        # 2) 拷贝数据到输入 Tensor
        # asnumpy() 返回与 Tensor 共享内存的 view，可直接写入
        np.copyto(self.in_tensor.asnumpy(), host_in)

        # 3) 推理执行
        self.engine.process(
            self.graph_name,
            {self.input_name:  self.in_tensor},
            {self.output_name: self.out_tensor}:
        )

        # 4) 拷回结果
        scores = self.out_tensor.asnumpy().copy()   # copy() 防止后续被覆盖
        return scores.squeeze()                     # 去掉 batch 维
