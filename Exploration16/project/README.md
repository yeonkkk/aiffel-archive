# E-16. 흐린 사진을 선명하게

`Super Resolution`을 사용하여 저해상도의 이미지를 고해상도의 이미지로 변환한다.  

`GAN`은 정밀한 고해상도 이미지를 생성하기에 효과적이지만 시간이 오래걸린다는 특징이 있다.  

<br><br>

## Super Resolution
> Super Resolution(초해상화): 저해상도 영상을 고해상도 영상으로 변환하는 작업

<br>

`픽셀`: 디스플레이를 구성하는 가장 작은 단위  

`RGB`: 빛의 3원색을 혼합하여 색을 나타내는 방식  

`해상도`: 픽셀의 개수가 많을수록 선명해진다(고해상도).

`CCTV`  해상도 문제, `의료 영상` 등에 효과적으로 사용될 수 있다.  

<br><br>

### Super Resolution 활용 시 문제점

- **ill-posed (inverse) problem**: 1개의 저해상도 이미지에 대해 다수의 고해상도 이미지가 나올 수 있는 점

- **super Resolution 문제의 복잡도**: 제한된 정보만을 이용해 많은 정보를 만들어내는 과정은 매우 복잡함 → 잘못된 정보 생성 가능성 증가

- **정량적 평가 척도**와 **사람의 시각적 관찰 평가**가 잘 일치하지 않음  

<br><br>

## Interpolation
> 보간법(interpolation): 값을 알고 있는 두 점 사이 지점의 값이 얼마일지를 추정하는 기법.
>  많은 딥러닝 기반 Super Resolution 연구에서 결과를 비교하기 위해 수행

<br>

`선형보간법(linear interpolation)`: 두 점 사이에 직선을 이용해 f(x)를 추정  
![image](https://user-images.githubusercontent.com/88660886/142337121-392ea204-aaab-42b6-8302-795f07505209.png)  
    [이미지 출처](https://bskyvision.com/789)

<br><br>

`삼차보간법(cubic interpolation)`: 3차(cubic) 함수를 활용하여 f(x)를 추정. 선형보간법과 달리 네 개의 점을 참조  
![image](https://user-images.githubusercontent.com/88660886/142337324-f25394a0-4e98-4491-b786-59d2c80ec379.png)  
    [이미지 출처](https://bskyvision.com/789)


`쌍선형보간법(bilinear interpolation)`: 선형보간법을 2차원으로 확장시킨 것. 4(=2x2)개의 점 참조   

`쌍삼차보간법(bicubic interpolation)`: 삼차보간법을 2차원으로 확장시킨 것. 16(=4x4)개의 점을 참조   

<br><br>

## SRCNN
> Super Resolution Convolutional Neural Networks.
> MSE(Mean Squared Error) loss function 사용

![image](https://user-images.githubusercontent.com/88660886/142338447-2bf6a5c8-602f-4ff5-9c78-6b79a8292e94.png)  
[이미지 출처](https://deepai.org/publication/deep-learning-for-single-image-super-resolution-a-brief-review)  

<br>

- 과정
  
  - Patch extraction and representation: 저해상도 이미지에서 patch 추출
  - Non-linear mapping: 다차원의 patch들을 non-linear하게 다른 다차원의 patch들로 매핑
  - Reconstruction: 다차원 patch들로부터 고해상도 이미지를 복원

<br><br>

## 이외 구조들

### VDSR (Very Deep Super Resolution)
- 저해상도 이미지의 크기를 늘려 입력으로 사용 (interpolation)
- 20개의 convolutional layer
- residual learning 이용: 고해상도 이미지 생성 직전 원본 이미지를 더함

<br><br>

### RDN (Residual Dense Network)
-  각 layer에서 나오는 출력을 최대한 활용 → 출력된 특징들을 이후에도 재활용

<br><br>

### RCAN (Residual Channel Attention Networks)
- 각각의 특징 맵을 대상으로 일부 중요한 채널에만 선택적으로 집중하도록 유도(Channel attention)

<br><br>

## 참고 자료

[모니터의 핵심, 디스플레이의 스펙 따라잡기](https://news.lgdisplay.com/kr/2014/03/%eb%aa%a8%eb%8b%88%ed%84%b0-%ed%95%b5%ec%8b%ac-%eb%94%94%ec%8a%a4%ed%94%8c%eb%a0%88%ec%9d%b4%ec%9d%98-%ec%8a%a4%ed%8e%99-%eb%94%b0%eb%9d%bc%ec%9e%a1%ea%b8%b0-%ed%95%b4%ec%83%81%eb%8f%84/)

[그림으로 쉽게 알아보는 HD 해상도의 차이](https://news.lgdisplay.com/kr/2014/07/%EA%B7%B8%EB%A6%BC%EC%9C%BC%EB%A1%9C-%EC%89%BD%EA%B2%8C-%EC%95%8C%EC%95%84%EB%B3%B4%EB%8A%94-hd-%ED%95%B4%EC%83%81%EB%8F%84%EC%9D%98-%EC%B0%A8%EC%9D%B4/)

[하얀거탑 리마스터링 제작기](http://tech.kobeta.com/%ED%95%98%EC%96%80%EA%B1%B0%ED%83%91-uhd-%EB%A6%AC%EB%A7%88%EC%8A%A4%ED%84%B0%EB%A7%81-%EC%A0%9C%EC%9E%91%EA%B8%B0/)

[Deep Learning for Single Image Super-Resolution:
A Brief Review](https://arxiv.org/pdf/1808.03344.pdf)

[선형보간법과 삼차보간법, 제대로 이해하자](https://bskyvision.com/789)

[ilinear interpolation 예제](https://blog.naver.com/dic1224/220882679460)

[OpenCV Documentation](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)

[논문리뷰 - SRCNN](https://d-tail.tistory.com/6)
