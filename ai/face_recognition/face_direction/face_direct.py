import time
import math
import onnxruntime
import numpy as np
import cv2
import torch


def get_num(point_dict, name, axis):
    num = point_dict.get(f'{name}')[axis]
    num = float(num)
    return num


def point_line(point, line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    k2 = -1.0 / k1
    b2 = y3 * 1.0 - x3 * k2 * 1.0
    x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def point_point(point_1, point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance


def cross_point(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


class FaceDirection:
    def __init__(self, direct_path, gpu_id=0):
        self.gpu_id = gpu_id
        self.input_height = 112
        self.input_width = 112
        if 'onnx' in direct_path:
            if gpu_id >= 0:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            self.direct = onnxruntime.InferenceSession(direct_path, providers=providers)
            self.backend = 'onnxruntime'
        elif 'engine' in direct_path:
            import tensorrt as trt
            from ai.torch2trt import TRTModule
            logger = trt.Logger(trt.Logger.INFO)
            with open(direct_path, 'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())

            input_names = []
            output_names = []
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i):
                    input_names.append(engine.get_binding_name(i))
                else:
                    output_names.append(engine.get_binding_name(i))
            self.session = TRTModule(engine, input_names=input_names, output_names=output_names)
            self.backend = 'tensorrt'

    def inference(self, srcimg):
        input_img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        input_img = (input_img.astype(np.float32) / 255.0 - 0.5) / 0.5
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
        if self.backend == 'onnxruntime':
            input_name = self.direct.get_inputs()[0].name
            outputs = self.direct.run(None, {input_name: input_img})
        elif self.backend == 'tensorrt':
            input = torch.from_numpy(input_img).to('cuda')
            with torch.no_grad():
                outputs = self.session(input)
            outputs = [output.cpu().numpy() for output in outputs]
        else:
            outputs = None
            print('no support')

        pre_landmark = outputs[1]
        pre_landmark = pre_landmark.reshape(-1, 2) * [112, 112]
        point_dict = {}
        i = 0
        for (x, y) in pre_landmark.astype(np.float32):
            point_dict[f'{i}'] = [x, y]
            i += 1

        # yaw
        point1 = [get_num(point_dict, 1, 0), get_num(point_dict, 1, 1)]
        point31 = [get_num(point_dict, 31, 0), get_num(point_dict, 31, 1)]
        point51 = [get_num(point_dict, 51, 0), get_num(point_dict, 51, 1)]
        crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
        yaw_mean = point_point(point1, point31) / 2
        yaw_right = point_point(point1, crossover51)
        yaw = (yaw_mean - yaw_right) / yaw_mean
        yaw = int(yaw * 71.58 + 0.7037)
        # print('yaw:', yaw)

        # pitch
        pitch_dis = point_point(point51, crossover51)
        if point51[1] < crossover51[1]:
            pitch_dis = -pitch_dis
        pitch = int(1.497 * pitch_dis + 18.97)
        print('pitch:', pitch)

        # roll
        roll_tan = abs(get_num(point_dict, 60, 1) - get_num(point_dict, 72, 1)) / abs(
            get_num(point_dict, 60, 0) - get_num(point_dict, 72, 0))
        roll = math.atan(roll_tan)
        roll = math.degrees(roll)
        if get_num(point_dict, 60, 1) > get_num(point_dict, 72, 1):
            roll = -roll
        roll = int(roll)

        return yaw, pitch, roll
