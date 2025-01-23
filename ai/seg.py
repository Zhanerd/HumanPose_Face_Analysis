import argparse
import cv2
import numpy as np
import onnxruntime as ort
import time

# 绫诲瀹氫箟绫诲埆鏄犲皠鍏崇郴锛屼娇鐢ㄥ瓧鍏告牸寮?
CLASS_NAMES = {
    0: 'class_name1',  # 绫诲埆 0 鍚嶇О
    1: 'class_name2'  # 绫诲埆 1 鍚嶇О
    # 鍙互娣诲姞鏇村绫诲埆...
}

# 瀹氫箟绫诲埆瀵瑰簲鐨勯鑹诧紝鏍煎紡涓?(R, G, B)
CLASS_COLORS = {
    0: (255, 255, 0),  # 绫诲埆 0 鐨勯鑹蹭负闈掗粍鑹?
    1: (255, 0, 0)  # 绫诲埆 1 鐨勯鑹蹭负绾㈣壊
    # 鍙互涓哄叾浠栫被鍒寚瀹氶鑹?..
}


class YOLO11Seg:
    def __init__(self, onnx_model):
        # 鍒涘缓 Ort 鎺ㄧ悊浼氳瘽锛岄€夋嫨 CPU 鎴?GPU 鎻愪緵鑰?
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )
        # 鏍规嵁 ONNX 妯″瀷绫诲瀷閫夋嫨 Numpy 鏁版嵁绫诲瀷锛堟敮鎸?FP32 鍜?FP16锛?
        self.ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single

        # 鑾峰彇妯″瀷鐨勮緭鍏ュ搴﹀拰楂樺害锛圷OLO11-seg 鍙湁涓€涓緭鍏ワ級
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]

        # 鎵撳嵃妯″瀷鐨勮緭鍏ュ昂瀵?
        # print("YOLO11 馃殌 瀹炰緥鍒嗗壊 ONNXRuntime")
        # print("妯″瀷鍚嶇О锛?, onnx_model)
        # print(f"妯″瀷杈撳叆灏哄锛氬搴?= {self.model_width}, 楂樺害 = {self.model_height}")

        # 鍔犺浇绫诲埆鍚嶇О
        self.classes = CLASS_NAMES

        # 鍔犺浇绫诲埆瀵瑰簲鐨勯鑹?
        self.class_colors = CLASS_COLORS

    def get_color_for_class(self, class_id):
        return self.class_colors.get(class_id, (255, 255, 255))  # 濡傛灉娌℃湁鎵惧埌绫诲埆棰滆壊锛岃繑鍥炵櫧鑹?

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        瀹屾暣鐨勬帹鐞嗘祦绋嬶細棰勫鐞?-> 鎺ㄧ悊 -> 鍚庡鐞?
        Args:
            im0 (Numpy.ndarray): 鍘熷杈撳叆鍥惧儚
            conf_threshold (float): 缃俊搴﹂槇鍊?
            iou_threshold (float): NMS 涓殑 IoU 闃堝€?
            nm (int): 鎺╄啘鏁伴噺
        Returns:
            boxes (List): 杈圭晫妗嗗垪琛?
            segments (List): 鍒嗗壊鍖哄煙鍒楄〃
            masks (np.ndarray): [N, H, W] 杈撳嚭鎺╄啘
        """
        # 鍥惧儚棰勫鐞?
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)

        # ONNX 鎺ㄧ悊
        infer_time = time.time()
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})
        print('infer_time ', time.time() - infer_time)

        # 鍚庡鐞?
        post_time = time.time()
        boxes, segments, masks = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        print('post_time ', time.time() - post_time)
        return boxes, segments, masks

    def preprocess(self, img):
        """
        鍥惧儚棰勫鐞?
        Args:
            img (Numpy.ndarray): 杈撳叆鍥惧儚
        Returns:
            img_process (Numpy.ndarray): 澶勭悊鍚庣殑鍥惧儚
            ratio (tuple): 瀹介珮姣斾緥
            pad_w (float): 瀹藉害鐨勫～鍏?
            pad_h (float): 楂樺害鐨勫～鍏?
        """
        # 璋冩暣杈撳叆鍥惧儚澶у皬骞朵娇鐢?letterbox 濉厖
        shape = img.shape[:2]  # 鍘熷鍥惧儚澶у皬
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # 濉厖瀹介珮
        if shape[::-1] != new_unpad:  # 璋冩暣鍥惧儚澶у皬
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 杞崲锛欻WC -> CHW -> BGR 杞?RGB -> 闄や互 255 -> contiguous -> 娣诲姞缁村害
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        鎺ㄧ悊鍚庣殑缁撴灉鍚庡鐞?
        Args:
            preds (Numpy.ndarray): 鏉ヨ嚜 ONNX 鐨勬帹鐞嗙粨鏋?
            im0 (Numpy.ndarray): [h, w, c] 鍘熷杈撳叆鍥惧儚
            ratio (tuple): 瀹介珮姣斾緥
            pad_w (float): 瀹藉害鐨勫～鍏?
            pad_h (float): 楂樺害鐨勫～鍏?
            conf_threshold (float): 缃俊搴﹂槇鍊?
            iou_threshold (float): IoU 闃堝€?
            nm (int): 鎺╄啘鏁伴噺
        Returns:
            boxes (List): 杈圭晫妗嗗垪琛?
            segments (List): 鍒嗗壊鍖哄煙鍒楄〃
            masks (np.ndarray): 鎺╄啘鏁扮粍
        """
        x, protos = preds[0], preds[1]  # 鑾峰彇妯″瀷鐨勪袱涓緭鍑猴細棰勬祴鍜屽師鍨?

        # 杞崲缁村害
        x = np.einsum("bcn->bnc", x)

        # 缃俊搴﹁繃婊?
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # 鍚堝苟杈圭晫妗嗐€佺疆淇″害銆佺被鍒拰鎺╄啘
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        x = x[x[:, 5] == 0]

        NMS_time = time.time()
        # NMS 杩囨护
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]
        print('NMS_time ', time.time() - NMS_time)

        # 瑙ｆ瀽骞惰繑鍥炵粨鏋?
        if len(x) > 0:
            # 杈圭晫妗嗘牸寮忚浆鎹細浠?cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # 缂╂斁杈圭晫妗嗭紝浣垮叾涓庡師濮嬪浘鍍忓昂瀵稿尮閰?
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # 闄愬埗杈圭晫妗嗗湪鍥惧儚杈圭晫鍐?
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # 澶勭悊鎺╄啘
            mask_time = time.time()
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
            print('mask_time', time.time() - mask_time)
            # 灏嗘帺鑶滆浆鎹负鍒嗗壊鍖哄煙
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # 杩斿洖杈圭晫妗嗐€佸垎鍓插尯鍩熷拰鎺╄啘
        else:
            h, w = im0.shape[:2]
            return [], [], np.zeros((h, w), dtype=np.uint8)

    @staticmethod
    def masks2segments(masks):
        """
        灏嗘帺鑶滆浆鎹负鍒嗗壊鍖哄煙
        Args:
            masks (numpy.ndarray): 妯″瀷杈撳嚭鐨勬帺鑶滐紝褰㈢姸涓?(n, h, w)
        Returns:
            segments (List): 鍒嗗壊鍖哄煙鐨勫垪琛?
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # 鎵惧埌杞粨
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # 濡傛灉娌℃湁鎵惧埌鍒嗗壊鍖哄煙锛岃繑鍥炵┖鏁扮粍
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_maskV1(masks, boxes):
        """
        瑁佸壀鎺╄啘锛屼娇鍏朵笌杈圭晫妗嗗榻?
        Args:
            masks (Numpy.ndarray): [n, h, w] 鎺╄啘鏁扮粍
            boxes (Numpy.ndarray): [n, 4] 杈圭晫妗?
        Returns:
            (Numpy.ndarray): 瑁佸壀鍚庣殑鎺╄啘
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    @staticmethod
    def crop_mask(masks, boxes):
        """
        瑁佸壀鎺╄啘锛屼娇鍏朵笌杈圭晫妗嗗榻愶紙鍚戦噺鍖栫増鏈級

        Args:
        masks (Numpy.ndarray): [n, h, w] 鎺╄啘鏁扮粍
        boxes (Numpy.ndarray): [n, 4] 杈圭晫妗嗭紝鏍煎紡涓篬x1, y1, x2, y2]

        Returns:
        (Numpy.ndarray): 瑁佸壀鍚庣殑鎺╄啘
        """
        n, h, w = masks.shape

        # 鎵╁睍boxes鐨勭淮搴︿互鍖归厤masks鐨勫舰鐘?
        x1, y1, x2, y2 = boxes[:, 0][:, None, None], boxes[:, 1][:, None, None], boxes[:, 2][:, None, None], boxes[:,
                                                                                                             3][:, None,
                                                                                                             None]
        r = np.arange(w)[None, None, :]  # 琛岀储寮?
        c = np.arange(h)[None, :, None]  # 鍒楃储寮?

        # 鍒涘缓甯冨皵鏁扮粍鏍囪鍦ㄨ竟鐣屾鍐呯殑浣嶇疆
        in_box = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)

        # 杩斿洖瑁佸壀鍚庣殑鎺╄啘
        return masks * in_box

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        澶勭悊妯″瀷杈撳嚭鐨勬帺鑶?
        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w] 鎺╄啘鍘熷瀷
            masks_in (numpy.ndarray): [n, mask_dim] 鎺╄啘鏁伴噺
            bboxes (numpy.ndarray): 缂╂斁鍒板師濮嬪浘鍍忓昂瀵哥殑杈圭晫妗?
            im0_shape (tuple): 鍘熷杈撳叆鍥惧儚鐨勫昂瀵?(h,w,c)
        Returns:
            (numpy.ndarray): 澶勭悊鍚庣殑鎺╄啘
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # 灏嗘帺鑶滀粠 P3 灏哄缂╂斁鍒板師濮嬭緭鍏ュ浘鍍忓ぇ灏?
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW

        mat_time = time.time()
        masks = self.crop_mask(masks, bboxes)  # 瑁佸壀鎺╄啘
        print('crop_time', time.time() - mat_time)

        # masks = np.greater(masks, 0.5)
        # masks = np.max(masks, axis=0)
        # masks = np.array(masks*255,dtype=np.uint8)

        # print(masks.shape)
        # print(np.max(masks))
        return np.greater(masks, 0.5)  # 杩斿洖浜屽€煎寲鍚庣殑鎺╄啘

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        灏嗘帺鑶滅缉鏀捐嚦鍘熷鍥惧儚澶у皬
        Args:
            masks (np.ndarray): 缂╂斁鍜屽～鍏呭悗鐨勬帺鑶?
            im0_shape (tuple): 鍘熷鍥惧儚澶у皬
            ratio_pad (tuple): 濉厖涓庡師濮嬪浘鍍忕殑姣斾緥
        Returns:
            masks (np.ndarray): 缂╂斁鍚庣殑鎺╄啘
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # 璁＄畻姣斾緥
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # 姣斾緥
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # 濉厖
        else:
            pad = ratio_pad[1]

        # 璁＄畻鎺╄啘鐨勮竟鐣?
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" 搴旇鏄?2 鎴?3锛屼絾寰楀埌 {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # 浣跨敤 INTER_LINEAR 鎻掑€艰皟鏁村ぇ灏?
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True):
        """
        缁樺埗鍜屽彲瑙嗗寲缁撴灉
        Args:
            im (np.ndarray): 鍘熷鍥惧儚锛屽舰鐘朵负 [h, w, c]
            bboxes (numpy.ndarray): [n, 4]锛宯 鏄竟鐣屾鏁伴噺
            segments (List): 鍒嗗壊鍖哄煙鐨勫垪琛?
            vis (bool): 鏄惁浣跨敤 OpenCV 鏄剧ず鍥惧儚
            save (bool): 鏄惁淇濆瓨甯︽敞閲婄殑鍥惧儚
        Returns:
            None
        """
        # 鍒涘缓鍥惧儚鍓湰
        im_canvas = im.copy()

        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # 鑾峰彇绫诲埆瀵瑰簲鐨勯鑹?
            color = self.get_color_for_class(int(cls_))

            # 缁樺埗杞粨鍜屽～鍏呮帺鑶?
            # cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # 缁樺埗鐧借壊杈规
            cv2.fillPoly(im_canvas, np.int32([segment]), color)  # 浣跨敤绫诲埆瀵瑰簲鐨勯鑹插～鍏呭杈瑰舰

            # 缁樺埗杈圭晫妗?
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1, cv2.LINE_AA)
            # 鍦ㄥ浘鍍忎笂缁樺埗绫诲埆鍚嶇О鍜岀疆淇″害
            cv2.putText(im, f"{self.classes[cls_]}: {conf:.3f}", (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 灏嗗浘鍍忓拰缁樺埗鐨勫杈瑰舰娣峰悎
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # 鏄剧ず鍥惧儚
        if vis:
            ima = cv2.resize(im, (1280, 720))
            cv2.imshow("seg_result_picture", ima)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 淇濆瓨鍥惧儚
        if save:
            cv2.imwrite("seg_result_picture.jpg", im)


if __name__ == "__main__":
    # 鍒涘缓鍛戒护琛屽弬鏁拌В鏋愬櫒
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=r"yolo11n-seg.onnx", help="ONNX 妯″瀷璺緞")
    parser.add_argument("--source", type=str,
                        default="/home/hz/server/ydai/test/ldty/1730773801208_land_in.jpg")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    # 鍔犺浇妯″瀷
    model = YOLO11Seg(args.model)

    # 浣跨敤 OpenCV 璇诲彇鍥惧儚
    img = cv2.imread(args.source)

    # 妯″瀷鎺ㄧ悊
    time1 = time.time()
    boxes, segments, mask = model(img, conf_threshold=args.conf, iou_threshold=args.iou)
    print('mask shape', mask.shape, mask.max(axis=0), time.time() - time1)
    time1 = time.time()
    boxes, segments, mask = model(img, conf_threshold=args.conf, iou_threshold=args.iou)
    print('mask shape', mask.shape, mask.max(axis=0), time.time() - time1)
    # cv2.imshow('a',mask)
    # cv2.waitKey(0)

    # 濡傛灉妫€娴嬪埌鐩爣锛岀粯鍒惰竟鐣屾鍜屽垎鍓插尯鍩?
    if len(boxes) > 0:
        model.draw_and_visualize(img, boxes, segments, vis=False, save=True)