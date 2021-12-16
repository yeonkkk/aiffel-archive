# GD2. 잘 만든 Augmentation, 이미지 100장 안부럽다.
> **Data Augmentation**: 주어진 데이터셋을 다양한 방법으로 증강시켜(augment) 학습 데이터셋의 규모를 늘리는 방법

<br>

- `augmentation`의 장점
  - 데이터가 많아지면 과적합을 줄일 수 있다.
  - augmentation을 통해서 실제 입력값과 비슷한 데이터 분포를 만들어 낼 수 있다.
  - 노이즈에 잘 대응할 수 있다.

<br><br>

## Image Augmentation 기법
텐서플로우 페이지를 참고 하였습니다.  


<br><br>


### Flipping
> 좌우 또는 상하로 이미지를 대칭하는 기능

- 정확한 답이 존재하는 문제(detection, segmentation 등)에 적용할 경우 라벨도 flipping 해줘야 한다.

<br><br>
<br><br>

### Gray scale
>  3가지 채널을 가진 RGB 이미지를 하나의 채널을 가지게 하는 기능

<br><br>
<br><br>

### Saturation
> RGB 이미지를 HSV 이미지로 변경하고 S(saturation) 채널에 오프셋(offset)을 적용

- `HSV`: Hue(색조), Saturation(채도), Value(명도) 3가지 성분으로 표현  

- 이미지를 보다 선명하게 만든다.

- 적용 후 다시 RGB 색상 모델로 변경

<br><br>
<br><br>

### Brightness
> 밝기 조절  
> RGB 채널에서 값을 더해주면 밝아지고, 빼주면 어두워지는 성질을 이용하여 Brightness를 변경  

<br><br>
<br><br>

### Rotation
> 이미지의 각도 변환

- 90도 단위 변환 → 직사각형 형태가 유지, 이미지 크기 조절 시 바로 사용 가능

- 이 외의 경우 빈 영역이 생기게 된다.

<br><br>
<br><br>

### Center Crop
> 이미지 중앙을 기준으로 확대하는 방법

- 너무 작게 적용할 경우 라벨과 맞지 않는 상황이 발생할 수 있음


<br><br>
<br><br>

### 이외 augmentation
- Gaussian noise
- Contrast change
- Sharpen
- Affine transformation
- Padding
- Blurring


<br><br>
<br><br>

## 참고 자료

[Tensorflow: Data augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

[영상인식과 색상모델(Gray,RGB,HSV,YCbCr)](https://darkpgmr.tistory.com/66)




