# GD 4. Object Detection
`Object detection`: 이미지 내에서 물체의 위치, 종류를 찾아내는 것으로 자율주행, 얼굴 인식 등 다양한 범위에서 사용된다.  

<br><br>


## Object Localization
> **Object detection**: 클래스 분류(classification) + 위치 측정 (localization)

`Localization`: 이미지 내에 하나의 물체가 있을 때 그 물체의 위치를 특정하는 것

`Detection`: 다수의 물체가 존재할 때 존재 여부 파악, 위치 특정, 클래스 분류까지 수행하는 것

물체의 위치를 표현하는 방법으로 `Bounding Box`, `Object Mask` 등이 있다.  



<br><br>
<br><br>

## 바운딩 박스(Bounding Box)
> 이미지 내에서 물체의 위치를 사각형으로 정의하고 이를 꼭짓점의 좌표로 표현하는 방식  

- 좌표 표현 방식 2가지
  
  - 전체 이미지의 좌측 상단을 원점으로 정의. 바운딩 박스의 좌상단 좌표 + 우하단 좌표로 표현하는 방식
  - 바운딩 박스의 폭과 높이로 정의(절대 좌표를 정의하지 않음)


<br><br>
<br><br>

## Intersection over Union(IoU)
> localization 모델의 결과 평가를 위한 지표(metric)  
> 면적의 절대적인 값에 영향을 받지 않도록 두 개 박스(prediction, ground truth)의 차이를 상대적으로 평가하기 위한 방법  
> 교차하는 영역을 합친 영역으로 나눈 값  

- 정답과 예측값이 유사해질수록 IoU는 1에 가까워진다.  


<br><br>
<br><br>

###  Target Label
![image](https://user-images.githubusercontent.com/88660886/146684688-32894e51-e786-492e-bf88-95e2ffa38e5f.png)  
```
[출처: https://youtu.be/GSwYGkTfOKk]
```  

<br><br>

- `p_c`: 물체가 있을 확률
  
- `c_1`, `c_2`, `c_3`: 각 클래스 1, 2, 3에 속할 확률  

- `b_x`, `b_y`, `b_h`, `b_w`: 
  
  - 바운딩 박스를 정의하기 위한 4개의 노드
  - `b_x`, `b_y`: 바운딩 박스 좌측 상단의 점 좌표
  - `b_h`, `b_w`: 바운딩 박스 높이와 폭
  - 입력 이미지의 너비 w, 높이 h로 각각 Normalize된 상대적인 좌표와 높이/폭으로 표시


<br><br>
<br><br>

## 슬라이딩 윈도우(Sliding Window)
> 이미지를 적당한 크기의 영역으로 나누고 각 영역에 대해 Localization network를 반복 적용하는 방식

- 이미지에서 잘라내는 크기 = 윈도우 크기

- 동일한 윈도우 사이즈의 영역을 이동시키면서(sliding) 수행

- 컨볼루션(Convolution)의 커널이 슬라이딩하는 것과 유사

- 문제점: 연산량, 속도

  - 처리해야할 window 수가 증가하면 소요 시간도 함께 증가하는 단점

  - 물체의 크기가 다양해지면 단일 크기 window가 커버가 어려울 수 있음 (처리속도 증가)


<br><br>
<br><br>

## 컨볼루션(Convolution)
> Object localization을 병렬로 수행  
> sliding window 방법을 활용하나, 병렬적으로 동시에 진행하여 비교적 빠른 속도로 처리 가능  

![image](https://user-images.githubusercontent.com/88660886/146685073-343b5c18-c260-4dad-b4be-6a8a68cfcf64.png)  
```
[출처: https://medium.com/datadriveninvestor/evolution-of-object-recognition-algorithms-i-5803c7be0691]
```


<br><br>
<br><br>

## 앵커 박스(Anchor box)
> 앵커박스는 서로 다른 형태의 물체가 겹쳐져 있는 경우 활용할 수 있다.  

![image](https://user-images.githubusercontent.com/88660886/146685377-2aa6e62f-30c5-4ec8-beea-eddb44449417.png)  
```
[출처: http://datahacker.rs/deep-learning-anchor-boxes/]
```
- Anchor box #1, Anchor box #2는 각각 다른 object을 위해 설정한 크기
- 앵커 박스가 2개가 됨에 따라 output dimension도 두 배가 된다.
- 인식 범위 내에 물체가 있고 두 개의 앵커 박스가 있는 경우 IoU가 더 높은 앵커 박스에 물체를 할당

<br><br>
<br><br>

## 바운딩 박스와 앵커 박스

|**바운딩 박스**|**앵커 박스**|
|:---|:---|
|네트워크가 predict한 object의 위치가 표현된 박스 |네트워크가 detect해야 할 object의 shape에 대한 가정(assumption)|
|네트워크의 출력|네트워크의 입력|



<br><br>
<br><br>

## NMS(Non-Max Suppression)
> 겹쳐진  여러 개의 박스를 하나로 줄여줄 수 있는 방법  
> 겹쳐진 박스가 있을 때 가장 확률이 높은 박스를 기준으로 IoU 이상인 것들을 삭제  

![image](https://user-images.githubusercontent.com/88660886/146685559-72929343-c770-4817-8d9a-d45b4347a8f1.png)  
```
[출처: https://www.quora.com/How-does-non-maximum-suppression-work-in-object-detection]
```



<br><br>
<br><br>

## Detection Architecture
![image](https://user-images.githubusercontent.com/88660886/146685989-2bb9cfac-e73b-4549-894b-6488d0bd859b.png)  
```
[출처: https://hoya012.github.io/blog/Tutorials-of-Object-Detection-Using-Deep-Learning-what-is-object-detection/]  
```

 <br><br>

![image](https://user-images.githubusercontent.com/88660886/146685867-533ab501-2a34-4f46-b828-4ac2aba4c042.png)  
```
[출처: https://medium.com/@jitender_phogat/1-2-introducing-retinanet-and-focal-loss-for-dense-object-detection-7ef9c4901b61]  
```

- Many stage Detector: 단계를 구분하여 수행하는 방식

  - 물체가 있을 법한 위치의 후보(proposals) 들을 뽑아내는 단계
  - Classification + 정확한 바운딩 박스를 구하는 Regression을 수행하는 단계

- One stage Detector 
  - 객체의 검출, 분류, 그리고 바운딩 박스 regression을 한 번에 하는 방법  
  - 상대적으로 처리속도가 빠름

<br><br>
<br><br>

##  Two-Stage Detector
> 대표적으로 `Faster-RCNN`가 있음   

<br><br>

### R-CNN
> region proposal을 selective search로 수행 후 컨볼루션 연산 수행  
> 한 이미지에서 특성을 반복 추출하기 때문에 비효율적이고 느린 단점

![image](https://user-images.githubusercontent.com/88660886/146686173-fc635e5a-5d34-4bd1-b8ac-4041f8618ddd.png)  
```
[출처: https://arxiv.org/pdf/1504.08083.pdf]  
```


<br><br>
<br><br>

### Fast R-CNN
> 후보 영역의 classification과 바운딩 박스 regression을 위한 특성을 한 번에 추출하여 사용  

![image](https://user-images.githubusercontent.com/88660886/146686280-11a5ded5-89e1-4932-84ca-a765776bd47a.png)  
```
[출처: https://jamiekang.github.io/2017/05/28/faster-r-cnn/]  
```

 - Sliding Window 방식 X
 - CNN을 거친 특성 맵(Feature Map)에 투영해, 특성 맵을 잘라냄
 - 한 번의 CNN을 거쳐 결과물을 재활용할 수 있어 연산량이 감소
 - `RoI(Region of Interest) pooling` 제안: 후보 영역에 해당하는 특성을 원하는 크기가 되도록 pooling
 - region proposal 알고리즘 병목 문제가 있음  

<br><br>
<br><br>

### Faster R-CNN
> region proposal 과정에서 RPN(Region Proposal Network)라고 불리는 신경망 네트워크를 사용  

![image](https://user-images.githubusercontent.com/88660886/146686301-cf5701b8-1580-4f1f-b181-170d8b67252c.png)  
```
[출처: https://arxiv.org/pdf/1506.01497.pdf]  
```

- 이미지에 CNN을 적용해 특성 추출 → 특성 맵으로 물체 존재 여부 판별 가능
- `RPN`: 특성 맵을 통해 후보 영역들을 얻어내는 네트워크

<br><br>
<br><br>

## One-Stage Detector
> 대표적으로 YOLO, SSD가 있음  



<br><br>

### YOLO (You Only Look Once)
> YOLO의 방식    
> 이미지를 그리드로 나눔  
> → 슬라이딩 윈도 기법을 컨볼루션 연산으로 대체  
> → Fully Convolutional Network 연산  
> → 그리드 셀 별로 바운딩 박스를 얻어냄  
> → 바운딩 박스들에 대해 NMS

![image](https://user-images.githubusercontent.com/88660886/146686502-96a2f16e-8e14-467f-8ae2-f4b1cc8fd0ee.png)  
```
[출처: https://arxiv.org/pdf/1506.02640.pdf]  
```

- 그리드 셀마다 클래스를 구분하는 방식 → 두 가지 클래스가 한 셀에 나타나는 경우 정확하게 동작하지 않음
- 빠른 인식 속도를 가짐
- 작은 물체를 잡기에 적합하지 않음

<br><br>
<br><br>

### SSD (Single-Shot Multibox Detector)
> 다양한 크기의 특성 맵을 활용하고자 한 방식  

![image](https://user-images.githubusercontent.com/88660886/146686669-dba78d16-9d4b-461f-b6cc-4414be3384bb.png)  
```
[출처: https://arxiv.org/pdf/1512.02325.pdf]  
```
- 다양한 크기의 특성 맵으로부터 classification과 바운딩 박스 regression을 수행

- 다양한 크기의 물체에 대응할 수 있는 detection 네트워크 생성  

-  여러 Feature map에서 detection을 위한 classification과 regression을 수행   
    → 앞단에서는 Low level Feature를 활용하여 작은 물체를 잡아낼 수 있고, 뒷단에서는 더 큰 영역을 볼 수 있다.

<br><br>
<br><br>


## 참고 자료

[딥러닝 객체 검출 용어 정리](https://light-tree.tistory.com/75)

[Review of Deep Learning Algorithms for Object Detection](https://medium.com/zylapp/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852)

[이미지 인식 문제의 개요(Sualab Blog)](https://sualab.github.io/introduction/2017/11/29/image-recognition-overview-2.html)

[C4W3L01 Object Localization](https://www.youtube.com/watch?v=GSwYGkTfOKk)

[C4W3L03 Object Detection](https://www.youtube.com/watch?v=5e5pjeojznk)

[라온피플 머신러닝 아카데미 - Fully Convolution Network](https://m.blog.naver.com/laonple/220958109081)

[C4W3L04 Convolutional Implementation Sliding Windows](https://www.youtube.com/watch?v=XdsmlBGOK-k)

[C4W3L08 Anchor Boxes](https://www.youtube.com/watch?v=RTlwl2bv0Tg&t=1s)

[C4W3L07 Nonmax Suppression](https://www.youtube.com/watch?v=VAo84c1hQX8)

[R-CNNs Tutorial](https://blog.lunit.io/2017/06/01/r-cnns-tutorial/)

[C4W3L10 Region Proposals](https://www.youtube.com/watch?v=6ykvU9WuIws)

[YOLO, Object Detection Network](https://blog.naver.com/PostView.nhn?isHttpsRedirect=true&blogId=sogangori&logNo=220993971883)

[curt-park님의 YOLO 분석](https://curt-park.github.io/2017-03-26/yolo/)

[C4W3L09 YOLO Algorithm](https://www.youtube.com/watch?v=9s_FpMpdYW8&t=1s)

[yeomko님의 갈아먹는 Object Detection 6 SSD: SIngle Shot Multibox Detector](https://yeomko.tistory.com/20)



