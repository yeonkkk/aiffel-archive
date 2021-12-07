# E-19. 난 스케치를 할 테니 너는 채색을 하거라
> 조건 없는 생성모델(unconditioned generative model)은 생성하고자 하는 데이터에 대한 제어가 어려움  

<br><br>


## GAN
>  Generator와 Discriminator 신경망이 minimax game을 통해 서로 경쟁하며 발전하는 구조.  
조건 없는 생성모델(unconditioned generative model)  

<br><br>

![](https://images.velog.io/images/tjddus0302/post/28a4bbac-a06c-4b9d-bae1-fead7667d710/image.png)

```
출처: AIFFEL EXPLORATION_SSAC2 19. 난 스케치를 할 테니 너는 채색을 하거라  
```

<br><br>

#### Generator
- 노이즈 z 입력 → 특정 representation(검정색) 변환 → 가짜 데이터 G(z) 생성   

#### Discriminator
- 실제 데이터 x와 가짜 데이터 G(z)를 입력 → D(x), D(G(z)) (보라색) 계산 (진짜, 가짜 식별)    


<br><br>
<br><br>

### GAN의 목적 함수
![CodeCogsEqn (1)](https://user-images.githubusercontent.com/88660886/145061774-a07470e5-f6ea-4841-9c41-ac1811b16322.png)


<!-- $$
{min_G}{max_D} {V(D,G)}=\mathbb{E}_{x\sim p_{data}~(x)}[log D(x)] + \mathbb{E}_{z\sim p_x(z)}[log(1-D(G(z)))]
$$    -->

<br><br>
<br><br>


## cGAN(Conditional Generative Adversarial Nets)
> GAN이 가진 생성 과정의 불편함을 해소.  
> 원하는 종류의 이미지를 생성하도록 고안된 방법  

<br><br>

![](https://images.velog.io/images/tjddus0302/post/0f5dd9ef-4c57-43f5-ae03-5a22f9a2f865/image.png)

```
출처: AIFFEL EXPLORATION_SSAC2 19. 난 스케치를 할 테니 너는 채색을 하거라  
```

<br><br>

#### Generator  
- z와 추가 정보 y 입력받음 → Generator 내부에서 결합 → representation(검정색)으로 변환 → 가짜 데이터 G(z∣y) 생성  

- MNIST나 CIFAR-10 등의 데이터셋에 대해 학습시키는 경우 y는 레이블 정보, 일반적으로 one-hot 벡터를 입력으로 넣음  


#### Discriminator
- x와 G(z∣y) 각각 입력받음 → y 정보가 함께 입력 → 진짜와 가짜를 식별  

- MNIST나 CIFAR-10 등의 데이터셋에 대해 학습시키는 경우 x와 y는 알맞은 한 쌍("7"이라 쓰인 이미지의 경우 레이블도 7)을 이뤄야 함.  


<br><br>
<br><br>

### cGAN의 목적 함수
![CodeCogsEqn (3)](https://user-images.githubusercontent.com/88660886/145063109-b31a1ebd-4bf6-44d2-b927-8a5d1f9a51ff.png)
<!-- $$
{min_G}{max_D} {V(D,G)}=\mathbb{E}_{x\sim p_{data}~(x)}[log D(x)] + \mathbb{E}_{z\sim p_x(x∣y)}[log(1-D(G(x∣y)))]
$$
 -->
- G와 D의 입력에 특정 조건을 나타내는 정보인 y를 같이 입력  

- y는 임의 노이즈 입력인 z의 가이드와 같음  

<br><br>
<br><br>

## Pix2Pix
> 이미지를 입력으로 하여 원하는 다른 형태의 이미지로 변환시킬 수 있는 GAN 모델  

- Conditional Adversarial Networks로 Image-to-Image Translation을 수행    

-  GAN 기반의 Image-to-Image Translation 작업에서 가장 기초가 되는 연구  

<br><br>
<br><br>

### Pix2Pix Generator

![](https://images.velog.io/images/tjddus0302/post/afbbc616-bf0e-4fe5-96d0-0e38420b2c41/image.png)

<br><br>

-  입력 이미지와 변환된 이미지의 크기는 동일해야 함  

- Encoder: 입력 이미지(x) 받음 → 단계적으로 이미지 down-sampling → representation 학습  
 
- Decoder: 이미지 up-sampling → 입력 이미지와 동일한 크기의 이미지(y) 생성  

- 위 과정은 모두 convolution 레이어로 진행  

- `bottleneck`  

	- Encoder의 최종 출력  
	
    	- 위 이미지 중간의 가장 작은 사각형  
    
   	 - 입력 이미지(x)의 가장 중요한 특징들을 담고 있음  

<br><br>
<br><br>

###  U-Net
![](https://images.velog.io/images/tjddus0302/post/3e6f6f42-00ea-40e3-a913-c7c863c62570/image.png)

<br><br>

- 레이어마다 Encoder, Decoder 연결(skip connection)되어 있음  

- Encoder로부터 더 많은 추가 정보를 얻는 방법


<br><br>
<br><br>

### Pix2Pix Loss Function

- `L1(MAE)` or `L2(MSE)` 손실만 이용하여 학습 → 결과 흐릿한 경향

- `L1+cGAN`:  L1손실과 GAN 손실을 같이 사용하여 더 좋은 결과 도출


<br><br>
<br><br>

### Pix2Pix Discriminator
![](https://images.velog.io/images/tjddus0302/post/3c6fca05-0c63-4430-ad57-3b2c89cc1ebe/image.png)
```
출처 : https://arxiv.org/pdf/1803.07422.pdf
```

#### PatchGAN
> 거리가 먼 두 픽셀은 서로 연관성이 거의 없기 때문에 특정 크기를 가진 일부 영역에 대해서 세부적으로 진짜/가짜를 판별

- Discriminator에 이미지 입력

- convolution 레이어를 거쳐 확률값을 나타내는 최종 결과 생성(여러 개의 값)

- 전체 영역이 아닌 일부 영역에 대해서만 진짜/가짜를 판별하는 하나의 확률값 도출

- 값을 평균하여 최종 Discriminator의 출력을 생성

- 너무 작은 patch는 품질이 떨어질 수 있음

<br><br>
<br><br>

## 참고 자료
[TF2-GAN](https://github.com/thisisiron/TF2-GAN)  

[[라온피플] Stochastic Pooling & Maxout
](https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=221259325819&proxyReferer=&proxyReferer=https:%2F%2Fwww.google.com%2F)  


[[Paper] Maxout Networks](https://arxiv.org/pdf/1302.4389.pdf)  

[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)  


[U-Net 논문 리뷰](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a)  


