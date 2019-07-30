# Tensorflow로 구현한 멀티쓰레드 object detection 프로그램
해당 프로젝트는 tensorflow 프레임워크를 사용하고 Faster RCNN, Mask RCNN, YOLO 모델들과 multi-threading을 사용해서 object 인식 속도를 높혔습니다. 
이 프로그램은 GPU(NVIDIA GeForce GTX 1080)에서 테스트가 되었습니다.해당 프로젝트에는 **트레이닝**을 하는 코드는 없습니다. 오로지 이미 학습된 모델들을 사용해서 object detection을 합니다. 
해당 프로그램은 object detection을 실행하기 전에 비디오의 frame수를 알고 시작해야 되기 때문에, real-time인 내장 카메라는 지원을 하지 않습니다. 이 프로그램은  
다음과 같이 작동을 합니다:

1. 비디오 파일을 읽는다
2. GPU의 숫자에 따라서 frame들을 나눈다
3. **COCO**에 pre-trained된 model을 multi-threading을 사용해서 동시에 object들을 감지합니다<br/>
ex) 2개의 GPU가 감지되었다면 전체 frame들을 반으로 나눕니다. 첫 번째 반은 gpu 0에서 object detection을 진행하고, 나머지 반은 gpu 1에서 진행합니다
4. 감지된 object에 사각형 박스와 라벨을 그립니다
5. Frame들을 다시 합치고 비디오를 저장합니다

구현을 하면서, 밑에 있는 레퍼지토리를 참고했습니다: 

* Tensorflow, Tensorflow-Yolo <br/>
https://github.com/tensorflow/models/tree/master/research/object_detection<br/> 
https://github.com/wizyoung/YOLOv3_TensorFlow<br/>


## 요구사항

* Python 3.6
* imutils 0.5.2<br> 
```pip install opencv-python imutils```
* tensorflow 1.14.0
* tensorflow-gpu 1.1.0<br>
```pip install tensorflow tensorflow-gpu```


## Pre-trained Models 다운 받는 방법
프로젝트에서는 2군데에서 다운 받은 pre-trained model들을 사용했습니다. 
### 1. Tensorflow Object Detection API

1.1 Tensorflow Object Detection API page 로 접속을 합니다
<br/>
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

<img src="tensorflow_api.png" width="600px"/>

1.2 이미지에 있는 표와 같은 표를 찾고 COCO로 학습된 pre-trained model을 다운 받습니다. 

### 2. Yolo
1.1 Yolo page에 접속을 합니다.<br/>
https://pjreddie.com/darknet/yolo/

1.2 Pre-trained model을 다운 받고, `darknet` 디렉토리 안으로 파일들을 옮깁니다.(해당 프로젝트는 Yolov3만 테스트를 했습니다)

## Yolo pre-trained model을 tensorflow에서 사용할 수 있게 변환시키는 방법
Pre-trained된 model들은 각자 고유의 형식을 가지고 있어서 바로 다른 framework에서 같은 파일을 사용하는 것이 안되는 경우가 많습니다. 그래서 다른 framework에서도 사용할 수 있도록
model의 형식을 변환하는 법을 알려드립니다.

### 1. Tensorflow Pre-trained model => Opencv Dnn
Tensorflow에서 작동하는 frozen inference graph `.pb` 형식의 model을 **Opencv**에서 사용하기 위해서는 해당 파일을 `.pbtxt` 형식으로 변환을 해야합니다. 
참고로, object detection 구조마다 `.pb`파일을 변환하는 법은 다릅니다.<br/>
예) 다운 받은 pre-trained model이 faster rcnn일 때, 
```Shell
$ python tf_text_graph_faster_rcnn.py --input /path/to/.pb --output /path/to/.pbtxt --config /path/to/pipeline.config
``` 

### 2. Yolo Pre-trained model => Tensorflow
Darknet에서 제공해주는 yolo의 `.weights` 형식의 모델을 tensorflow에서 사용하기 위해서는 `ckpt` 형식으로 변환을 해야합니다. 
첫 째로, `yolo_anchors.txt` 파일이 `darknet` 디렉토리 안에 있는지 확인을 합니다. 
둘 째로, 다운 받은 `yolov3.weights` 와 `yolov3.cfg` 파일들을 `darknet` 디렉토리 안으로 옮깁니다.<br/>
```Shell
python convert_weight.py
``` 
해당 프로그램을 실행시키면, `ckpt` 형식의 파일이 만들어지고 이 파일도 `darknet` 디렉토리 안에 있는 것을 확인합니다.

## Demo 실행
비디오 영상 파일이 `assets` 디렉토리 안, yolo `weights, ckpt, cfg` 파일들이 `darknet` 디렉토리 안에 있을 때의 demo를 실행시키는 방법입니다. 
추가로, 밑에 `/path/to/...` 이러한 부분이 나오면 해당 파일이 있는 경로로 대체를 해야합니다. 비디오 저장 default는 `False`이기 때문에 
비디오 저장을 원한다면 `--save True`를 argument에 추가해줘야 합니다.
#### FasterRCNN Tensorflow
```Shell
python tensorflow_pretrained_multithreading.py \
    --video assets/cars.mp4 \
    --frozen /path/to/frozen_inference_graph.pb \
    --conf 0.5 \
    --save True
```

#### Yolo Tensorflow
```Shell
python tensorflow_yolo_tensorflow_pretrained_multithreading.py \
    --video assets/cars.mp4 \
    --ckpt darknet/yolov3.ckpt \
    --conf 0.5 --nms 0.4 \
    --anchor_path darknet/yolo_anchors.txt \
    --save True
```
두개의 GPU를 사용해서 동시에 object detection을 진행한 결과물이다.동시에 detection이 진행되지만 하나의 비디오로 저장된다.
<div align="center">
<img src="readme/multithread_cars.gif" width="600px"/>
</div>
<div align="center">
<img src="readme/multithread_image.png"/>
</div>


### Credits:
Video by Pixabay from Pexels<br/>
Video by clara blasco from Pexels <br/>
Video by Pixly Videos from Pexels<br/>
Video by George Morina from Pexels <br/>