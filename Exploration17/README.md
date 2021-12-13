# E-17. 인간보다 퀴즈를 잘푸는 인공지능

- **BERT(Bidirectional Encoder Representations from Transformers)**

  - 엄청난 규모의 언어 모델을 사전학습 
  
  - 제너럴한 방식(어떤 태스크에든 일부 fine-tuning만을 통해 손쉽게 적용하여 해결)이 가능함을 입증

<br><br>
<br><br>

## KorQuAD(The Korean Question Answering Dataset)
>  한국어 질의응답 데이터셋   
>  미국 스탠퍼드 대학에서 구축한 대용량 데이터셋인 SQuAD 벤치마킹  

- 3가지 척도로 모델 평가
  
  - `EM`(Exact Match): 모델이 정답을 정확히 맞춘 비율
  - `F1 score`: 모델이 낸 답안과 정답이 음절 단위로 겹치는 부분을 고려한 부분점수
  - `1-example-latency`: 질문당 응답속도

<br><br>
<br><br>

###  KorQuAD1.0과 2.0
> 문서의 길이, 구조 그리고 답변의 길이와 구조에서 차이가 있다.

<br><br>
<br><br>

## 워드 클라우드(Word Cloud)
> 자료의 빈도수를 시각화해서 나타내는 방법

- 문서의 핵심 단어를 한눈에 파악 가능

- 빅데이터를 분석할 때 데이터의 특징을 도출하기 위해서 활용

- 빈도수가 높은 단어일수록 글씨 크기가 큼

<br><br>
<br><br>
## BERT 모델 구조
![image](https://user-images.githubusercontent.com/88660886/144016153-19d5e6f4-bd75-4584-b7f4-6fbc5ca828ee.png)  
```
출처: AIFFEL EXPLORATION_SSAC2 17. 인간보다 퀴즈를 잘푸는 인공지능
```

Transformer Encoder 구조 활용  

기존 tramsformer와 차이점: Layer 개수는 12개 이상으로 늘림, 파라미터 크기 커짐 (기본적인 구조는 동일)  

Transformer Encoder에 넣었을 때, 출력 모델이 `Mask LM`, `NSP` 라는 2가지 문제를 해결하도록 구성  

<br><br>
<br><br>

### Mask LM(Masked Language Model) 
>  다음 빈칸에 알맞은 말은 문제를 대량으로 풀어보는 언어 모델

<br><br>

![image](https://user-images.githubusercontent.com/88660886/144017331-0f456561-80fd-4d7d-a0cb-24e234e85a1f.png)
```
출처: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
```
-  input에서 무작위하게 몇개의 token을 mask 

- Transformer 구조에 넣어서 주변 단어의 context만을 보고 mask된 단어를 예측


<br><br>
<br><br>

### Next Sentence Prediction
> `<SEP>`를 경계로 좌우 두 문장이 순서대로 이어지는 문장이 맞는지를 맞추는 문제

![image](https://user-images.githubusercontent.com/88660886/144018036-9b4256cf-8274-447a-a0b2-9e71c2990784.png)

```
출처: AIFFEL EXPLORATION_SSAC2 17. 인간보다 퀴즈를 잘푸는 인공지능
```

<br><br>
<br><br>

### Token Embedding
>  `tokenizer`로 `Word Piece model`이라는 `subword tokenizer`를 사용

빈도가 높은 긴 길이의 subword도 하나의 단위로 만듦  

빈도가 낮은 단어는 다시 subword 단위로 쪼갠다.  

빈도가 낮은 단어가 OOV(Out-of-vocabulary) 처리되는 것을 방지해 주는 장점  

<br><br>
<br><br>

### Segment Embedding
> 각 단어가 어느 문장에 포함되는지 역할을 규정

단어가 Question 문장에 속하는지, Context 문장에 속하는지와 같은 구분이 필요한 경우 유용  

<br><br>
<br><br>

### Position Embedding
> Transformer에서 사용되던 position embedding과 동일  


<br><br>
<br><br>

## 참고 자료
[BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

[SQuAD 공식홈페이지](https://rajpurkar.github.io/SQuAD-explorer/)

[KorQuAD 공식홈페이지](https://korquad.github.io/)

[MRC 모델, 어떻게 개발하고 평가하나요?](https://blog.naver.com/skelterlabs/222025030327)

[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation
](https://arxiv.org/pdf/1609.08144.pdf)

[SentencePiece 모델](https://github.com/google/sentencepiece)

[BERT 논문정리](https://tmaxai.github.io/post/BERT/)
