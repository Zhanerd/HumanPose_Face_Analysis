# HumanPose_Face_Analysis
This project contain face,huamn-pose detect(yolo,rtmdet,etc) and recognition(arcface,rtmpose,etc). 
Try to provide inference on onnx,tensorRT,rknn...  
# Update Log
2025/04/03  U can download onnx models by google dirver. See details in ai/models/readme.  
2025/01/23  Add some new feature(Ocr,TTS,YOLO_obb). (Warning, this tts model is only support chinese! | TTS目前仅仅支持中文，且encoder还不能转化为engine，静候更新x。X)  
2024/10/31  Support Onnx and tensorRT(8.6.1) inference.
# Requirements
    python>=3.8
    torch==2.1.1
    onnxruntime-gpu==1.18.0
    tensorrt==8.6.1
In this project, u dont need to install insightface or ultralyitcs or paddlepaddle or mmpose!
# USAGE
There will briefly introduce how to use it.
## Face
Face, check face_reco.py. See the detail of the class FaceRecognition. And '__main__' is the demo. Also provide single choose for face det,recognition,direction,quality(see the file for how to use).  

Model ### use face_det_10g for face detection;use face_w600k_r50 for face feature extraction;use face_quality_assessment for face quality assessment.    
Usage ### 

    # init model
    det_path="your_path_face_det_10g"
    reg_path="your_path_face_w600k_r50"
    quality_path="your_path_face_quality_assessment"
    face_recognitio = FaceRecognition(det_path=det_path,
                                      reg_path=reg_path,
                                      quality_path = quality_path,
                                      gpu_id=0)
    # init data
    group_embedding = np.load("your_path_group_embedding")
    group_id = np.load("your_path_group_id")
    # detect face
    frame = cv2.imread("img_path")
    results = face_recognitio.detect(frame, quality=False)  
    '''
    results is a list of dict, each dict is a face, keys are follows:
    bbox is a int32 numpy array of [x1,y1,x2,y2]
    kps is a int32 numpy array of [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5], representing the five facial landmarks
    embedding is a float32 numpy array of 1x512
    when quality is True, error_code and error_message will be returned, detail see the file
    '''
    # match face
    for result in results:
        # frame = cv2.rectangle(frame, (result['bbox'][0], result['bbox'][1]), (result['bbox'][2], result['bbox'][3]),(0, 0, 255), 2) ### if u wanna draw the face box, u can use this code 
        in_embedding.append(result['embedding'])
        match_info = face_recognitio.match_feature(result['embedding'], group_embedding, group_id, thre=0.7)
    '''
    match_info is a list of dict, each dict is a person, keys are follows:
    person_id is a string, refer to group_id.
    similarity is a float, show the max similarity.
    '''
## Pose
Pose, check mmpose.py. See the detail of the class TopDownEstimation. And '__main__' is the demo. 
Also provide single choose for det(yolo,rtmdet),pose(rtmpose),track(deepsort)
## Segment
Beside, there are some segment methods for human, such as PSPNet,yolo-seg,detins. But it still need valid, so i didnt package it.
## Ocr
Ocr, check ppocr.py. See the detail of the class PaddlePaddleOcr. And '__main__' is the demo. 
## TTS
Tts, check t2s.py. See the detail of the class Text2Speech. And '__main__' is the demo. (Only cn)
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
PaddleOCR                   https://github.com/PaddlePaddle/PaddleOCR
