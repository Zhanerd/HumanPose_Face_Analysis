# HumanPose_Face_Analysis
This project contain face,huamn-pose detect(yolo,rtmdet,etc) and recognition(arcface,rtmpose,etc). 
Try to provide inference on onnx,tensorRT,rknn...  
# Update Log
2025/01/23  Add some new feature(Ocr,TTS,YOLO_obb). 

2024/10/31  Support Onnx and tensorRT(8.6.1) inference.
# Requirements
    python >= 3.8
    torch == 2.1.1
    onnxruntime-gpu == 1.18.0
    tensorrt == 8.6.1
In this project, u dont need to install insightface or ultralyitcs or paddlepaddle or mmpose!
# 
# USAGE
There will briefly introduce how to use it.
## Face
Face, check face_reco.py. See the detail of the class FaceRecognition. And '__main__' is the demo. 
Also provide single choose for face det,recognition,direction,quality(see the file for how to use).
## Pose
Pose, check mmpose.py. See the detail of the class TopDownEstimation. And '__main__' is the demo. 
Also provide single choose for det(yolo,rtmdet),pose(rtmpose),track(deepsort)
## Segment
Beside, there are some segment methods for human, such as PSPNet,yolo-seg,detins. But it still need valid, so i didnt package it.
## Ocr
Ocr, check ppocr.py. See the detail of the class PaddlePaddleOcr. And '__main__' is the demo. 
## TTS
Tts, check t2s.py. See the detail of the class Text2Speech. And '__main__' is the demo. 
## YOLO_obb
YOLO_obb, use yolo_obb.py. See the detail of the class YOLO_OBB. And '__main__' is the demo. 
# License
This repository is licensed under the Apache-2.0 License.
# Reference
Insightface                 https://github.com/deepinsight/insightface  
FaceImageQuality            https://github.com/pterhoer/FaceImageQuality  
MMpose                      https://github.com/open-mmlab/mmpose  
rtmlib                      https://github.com/Tau-J/rtmlib  
deep_sort_pytorch           https://github.com/ZQPei/deep_sort_pytorch
