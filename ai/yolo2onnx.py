from ultralytics import YOLO

model = YOLO(r'C:\Users\84728\Desktop\yolov8_11_e70.pt')


# 导出模型到ONNX格式
model.export(format='onnx')