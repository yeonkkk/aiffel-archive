# Face detection
이전에 활용했던 `Dlib 라이브러리`는 몇가지 문제점이 존재했다.
(얼굴 탐지가 잘 되지 않음, 느린 속도로 동영상 처리 어려움, 얼굴 각도나 방향과 같은 변화에 취약 등)

 <br><br>


`edge device(핸드폰)`에서 기능을 바로 활용하기 위해서는 `네트워크 비용`, `서버 비용`, `인터넷 속도`의 영향을 고려해야한다.   

 <br><br>


핸드폰에서 모델을 활용하려면 `weight`가 작은 모델이 좋지만, 성능이 떨어질 가능성이 크다.   
또한 빠른 반응 속도도 고려하여야한다. 

`face detection`에서 `sliding window`가 많은 시간을 차지하기 때문에 `2-stage 방식`을 후 순위로 두는 것을 생각해볼 수 있다.  
→  `Single Stage Object Detection` 을 사용하자!  

 <br><br>


빠른 속도를 위해서 연산의 `병렬화`도 가능해야한다.    

안드로이드는 `MLKit`, 아이폰은 `CoreML` 라이브러리를 통해 병렬화가 가능하다.  
`TFLite`를 이용하여 훈련된 모델을 사용할수도 있다.  

 <br><br>
 <br><br> 
 
 # YOLO v1 (You Only Look Once)
 >  YOLO v1에서 grid는 고정되고, 각 `grid 안에 물체가 있을 확률`이 중요함  
 >  YOLO의 목표는 grid에 포함되는 물체를 잘 잡아내는 것  
 >  (bbox의 ground truth와 최대한 동일하도록 학습되는 것)

 <br><br>

## 모델별 가정
 |RCNN|YOLO v1|
 |:---:|:---:|
 |물체가 존재할 것 같은 곳을 `backbone network`로 표현 가능|이미지 내의 `작은 영역`을 나누면 그곳에 물체 존재할수도 있음|
 |`region proposal network`|`grid` 내에 물체가 존재|
 
 <br><br>
 <br><br> 
 
 ## grid cell
 ![image](https://user-images.githubusercontent.com/88660886/149054469-f10e7350-6f94-4ebf-9d26-ff7e3d8036a5.png)   
 ```
 출처: AIFFEL Going Deeper(CV)_SSAC2 13. 멀리 있지만 괜찮아
 ```
- 이미지를 S x S grid로 나눔  

-  `grid cell`: `bounding box(bbox)`과 각 box의 `confidence score` 예측  
 
   -   `confidence score`: bbox가 사물(object) 포함하는 것을 확신하는 정도 & 예측 정확도 점수

-  각 bbox는 `x`, `y`, `w`, `h`, `confidence` 총 5개의 예측 
 
   -  `(x, y)` 좌표: bbox의 중심 좌표  
   -  `w`, `h`: 너비와 높이   

- 각 grid cell은 C개의 조건부 class 확률 `P(Class_i|Object)`도 예측  
 
  - `P(Class_i|Object)`: grid cell이 사물을 포함할 때를 조건으로 하는 확률  

- YOLO의 예측값: `S x S x (B * 5 + C) 크기`의 텐서로 출력   


 <br><br>
 <br><br> 
 
##  NMS(Non-Maximum Suppression)
> 비-최대 억제.   
> object detector가 예측한 bounding box 중 `정확한 bounding box를 선택`하는 기법  
 
 <br><br>
 <br><br> 
 
 ## YOLO 특징
 ![image](https://user-images.githubusercontent.com/88660886/149056415-5ba43e86-9144-4f7c-a66a-437e3290d25b.png)  
 ```
 출처: AIFFEL Going Deeper(CV)_SSAC2 13. 멀리 있지만 괜찮아
 ```
- 한 개의 grid당 bbox의 좌표와 confidence score만 예측 → R-CNN 계열 방법보다 빠른 속도

- 최초의 real-time object detector
 
 
 <br><br>
 <br><br> 
 
 ## Inference
 ![image](https://user-images.githubusercontent.com/88660886/149056670-304b1589-32a2-44b2-9e47-a64e7f0f9033.png)  
 ```
 출처: AIFFEL Going Deeper(CV)_SSAC2 13. 멀리 있지만 괜찮아
 ```
- 테스트 시 `P(Class_i|Object)`를 각 box의 `confidence prediction`과 곱해 `class-specific confidence score`를 얻을 수 있음
 
- score는 클래스 확률과 예측된 box가 사물에 얼마나 잘 맞는지를 나타냄
 
![image](https://user-images.githubusercontent.com/88660886/149057287-3c317d44-3b18-4b44-bb11-c5eead682f9b.png)

 <br><br>
 <br><br> 
 
## loss function
![image](https://user-images.githubusercontent.com/88660886/149057569-8cdb48db-4b0e-4920-b6f4-081f568efb3e.png)  

![image](https://user-images.githubusercontent.com/88660886/149057579-12ac10fc-f262-4142-9b84-935572f8fbca.png)  

 ```
 출처: AIFFEL Going Deeper(CV)_SSAC2 13. 멀리 있지만 괜찮아
 ```  
 
 <br><br>
 <br><br> 
 
 ## YOLO v1 단점
 - 각 grid cell이 하나의 클래스만 예측 가능 → 작은 객체 예측 어려움
 - bbox의 형태가 training data를 통해 학습 → bbox 분산 넓음 → 새로운 형태의 bbox 예측 어려움
 -  backbone만 거친 feature map을 대상으로 bbox 정보를 예측 → localization 부정확
 
 
 <br><br>
 <br><br> 
 
 # YOLO v2
 > 목적: `Make it better`, `Do it faster`, `Makes us stronger`   
  
- Make it better: 정확도를 올리기 위한 방법  
    -  Batch Normalization
    -  High Resolution Classifier
    -  Convolutional with Anchor boxes
    -  Dimension Clusters
    -  Direct location prediction
    -  Fine-Grained Features
    -  Multi-Scale Training
 
  
 <br><br> 
 
 
 
- Do it faster: 속도를 향상시키기 위한 방법
  - Darknet-19
  - Training for classification
  - Training for detection
 
  
 <br><br> 
 
 
 
- Makes us stronger: 더 많은 범위의 class를 예측하기 위한 방법
  - Hierarchical classification
  - Dataset combination with WordTree
  - Joint classification and detection
 
 
 <br><br>
 <br><br> 
 
 # SSD (Single Shot MultiBox Detector)
 >  YOLO v1에서 grid를 사용해서 생기는 단점에 대한 해결책 제시   
 >  → `Image Pyramid`, `Pre-defined Anchor Box`

 
 <br><br>
 <br><br> 
 
## Image Pyramid
> ImageNet으로 사전학습된 VGG16 사용  
> 다양한 크기의 feature map을 사용(38x38, 19x19, 10x10, 5x5, 3x3 등)   
> → 원본 이미지에서 grid 크기를 다르게 하는 효과  

![image](https://user-images.githubusercontent.com/88660886/149059166-f5d991d9-9db3-4e3e-b31d-69be10b0161b.png)  

 ```
 출처: https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html#loss-function
 ```  

- 단점: feature map이 많아 계산량도 많음, 38x38은 box를 계산하기에 충분히 깊지 않을 수 있음  

  
 <br><br>
 <br><br> 
 
## SSD Workflow
> YOLO v1의 box 예측을 위한 seed 정보가 없어 넓은 bbox 분포를 모두 학습해야하는 단점(성능 손실) 보완을 위해 Faster R-CNN 등에서 사용하는 anchor 적용   
> → `Default box`: SSD의 anchor box 
> (pre-defined된 box의 x, y, w, h를 refinement하는 layer를 추가)  
 
  
 <br><br>
 <br><br> 
 
 ## SSD Loss function
 > 3개의 Loss function (Objective Loss Function, Localization Loss Function, Confidence Loss Function)
 
   
 <br><br>
 
- x:category p에 대한 i번째 Default box와 j번째 Ground Truth box의 물체 인식 지표. 
    (0.5 이상이면 1, 미만이면 0으로 정의)
    
- N: 매치된 Default box의 개수, N이 0이면 loss는 0

- l: 예측된 상자(Predicted box)

- g: Ground Truth box

- d: Default bounding box

- cx,cy: Default bounding box의 x, y 좌표

- w,h: Default bounding box의 width, height

- α: 교차 검증으로 얻어진 값 (α=1) 
  
  
 <br><br>
 <br><br> 
 
 ### Objective Loss Function
 > Localization Loss(loc)와 Confidence Loss(conf)의 가중합(weighted sum)
 
 ![image](https://user-images.githubusercontent.com/88660886/149060158-6bc7b175-d99b-4d80-9730-411c99b27d38.png)  

  
 <br><br>
 <br><br> 
 
 ### Localization Loss Function
 > 예측된 박스 l과 Ground truth box g 파라미터 사이의 Smooth L1 loss
 
 ![image](https://user-images.githubusercontent.com/88660886/149060213-d9ca2115-7f79-4ac2-bc28-9799aa3feb41.png)
 
-  예측 값(`g^`)의 cx, cy는 Default box의 cx, cy, w, hcx,cy,w,h로 normalize 된다.

- 이미 IoU가 0.5 이상된 부분에서만 고려 
   - 상대적으로 크지 않은 값들을 예측
   - 비교적 빠르게 수렴할 것으로 예상 가능

- 초기 값과 예측값(`g^`)의 w, h는 Default box에서 시작
- 예측된 l값들은 box를 표현할 때마다 default box의 offset 정보가 필요
  
 <br><br>
 <br><br> 
 
 ### Confidence Loss Function
 > 여러 매칭된(Positive) class에 대해 softmax를 취해준다.  
 > 매칭되지 않은(Negative) class를 예측하는 값은 배경이면 1, 아니면 0의 값을 가짐  
 > 최종 predicted class score: 예측할 class + 배경 class를 나타내는 지표  
 
 ![image](https://user-images.githubusercontent.com/88660886/149060335-4d6cd3e5-eca8-4b5d-9342-6e29636f90b8.png)
 
 <br><br>
 <br><br> 
 
 ### Hard negative mining
대부분의 Default box가 배경이기 때문에 마지막 class의 loss 부분에서는 positive:negative 비율을 1:3으로 정해 출력  
high confidence 순으로 정렬해 상위만 사용
  
 <br><br>
 <br><br> 
 
 ### FCOS
 > 기존의 anchor box기반에서 벗어나 pixelwise로 예측  
 > Anchor box 부작용을 해결 및 좋은 성능을 보임  
 > (학습 계산량, 하이퍼파리미터에 민감한 성능 등)
 
 <br><br>
 <br><br> 
 
 ## 참고 자료
 
 [카카오 얼굴인식 관련 리서치](https://tech.kakaoenterprise.com/63)
 
 [네이버 얼굴검출 관련 오픈소스](https://github.com/clovaai/EXTD_Pytorch) 
 
 
 [ore ML](https://developer.apple.com/documentation/coreml) 
 
 
 [ios 11 machine learning for everyone](https://machinethink.net/blog/ios-11-machine-learning-for-everyone/) 
 
 
 [ML kit](https://www.slideshare.net/inureyes/ml-kit-machine-learning-sdk) 
 
 
 [TFLite](https://www.tensorflow.org/lite?hl=ko) 
 
 [SIMD-병렬-프로그래밍](https://stonzeteam.github.io/SIMD-%EB%B3%91%EB%A0%AC-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D/) 
 
 [OpenCL](https://www.khronos.org/opencl/) 
 
 [Android: Open GL ES](https://developer.android.com/guide/topics/graphics/opengl?hl=ko)
 
 [Fast Detection Models](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html)  
 
 [What do we learn from single shot object detectors, FPN & Focal loss?](https://jonathan-hui.medium.com/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d)  
 
 [What do we learn from single shot object detectors, FPN & Focal loss? 번역](https://murra.tistory.com/17)  
 
 [You Only Look Once: Unified, Real-Time Object Detection](https://youtu.be/NM6lrxy0bxs)  
 
 [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)  
 
 [Non-Maximum Suppression(NMS)란](https://naknaklee.github.io/etc/2021/03/08/NMS/)  
 
 [IoU, Intersection over Union 개념을 이해하자](https://ballentain.tistory.com/12)  
 
 [Deep System’s YOLO](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_2158)  
 
[[Deeplearning] YOLO9000: Better, Faster, Stronger](https://dhhwang89.tistory.com/136)

[ YOLO9000: Better, Faster, Stronger](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)

[YOLOv3: An Incremental Improvement](https://taeu.github.io/paper/deeplearning-paper-yolov3/)

[YOLO v4](https://arxiv.org/pdf/2004.10934.pdf)


[ SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)


[SSD- Single Shot Multibox Detector](https://seongkyun.github.io/papers/2019/07/01/SSD/)

[FCOS:Fully Convolutional One-Stage Object Detectionan](https://arxiv.org/pdf/1904.01355.pdf)

["FCOS", One shot Anchor - Free Object Detection](https://blog.naver.com/jinyuri303/221876480557)


[S3FD- Single Shot Scale-invariant Face Detector (간단히)](https://seongkyun.github.io/papers/2019/03/21/S3FD/)

[S3FD: Single Shot Scale-invariant Face Detector
DSFD](https://arxiv.org/abs/1708.05237)

[DSFD: Dual Shot Face Detector
RetinaFace](https://arxiv.org/pdf/1810.10220.pdf)

 [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/pdf/1905.00641.pdf)
