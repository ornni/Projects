![title](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_title.PNG?raw=true)<br>

**목차**<br>

0. Abstract<br>

1. Introduction<br>

2. RNN Encoder-Decoder<br>

2.1 Preliminary: Recurrent Neural Networks<br>

2.2 RNN Encoder-Decoder<br>

2.3 Hidden Unit that Adaptively Remembers and Forgets<br>

3. Statistical Machine Translation<br>

3.1 Scoring Phrase Pairs with RNN Encoder-Decoder<br>

3.2 Related Approaches: Neural Networks in Machine Translation<br>

4. Experiments<br>

4.1 Data and Baseline System<br>

4.1.1 RNN Encoder-Decoder<br>

4.1.2 Neural Language Model<br>

4.2 Quantitative Analysis<br>

4.3 Qualitative Analysis<br>

4.4 Word and Phrase Representations<br>

5. Conclusion<br>

---

# 0. Abstract<br>

심층신경망은 이의제기 인식, 음성 인식에서 성공적<br>

= 자연어처리(NLP)에서 성공적<br>

통계적 기계번역(SMT)에서 성공적<br>

 #

SMT 위한 신경망 사용 연구라인을 따라,<br>

기존의 구문 기반 SMT system 일부로 사용될 수 있는 새로운 신경망 아키텍처에 초점<br>

 #

새로운 RNN Encoder-Decoder model 제안<br>

---

# 2. RNN Encoder-Decoder<br> 

## 2.1 Preliminary: Recurrent Neural Networks<br>
 

RNN은 hidden state h와 가변길이 sequence X=(X1, …, Xt)에서 작동하는 선택적 출력 y로 구성된 신경망<br>

각 시간 단계 t에서 hidden state h(t) update<br>

h(t)=f(h(t-1), xt)<br>

#

RNN은 sequence의 다음 symbol을 예측하도록 훈련되어 sequence에 대한 확률 분포 학습 가능<br>

이때 각 시간 단계 t의 출력은 조건부분포 p(xt |x(t-1), …, x1)<br>

![3-1](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_3-1.PNG?raw=true)<br>

#

## 2.2 RNN Encoder-Decoder
 

확률론적 관점에서 논문에서 제안한 모델은 또 다른 가변 길이 sequence를 조건으로 하는 가변길이 sequence에 대한 조건부분포를 학습하는 일반적인 방법(입력과 출력 sequence길이를 뜻하는 T와 T'의 길이는 다를 수 있음)<br>

p(y1, …, yT′ |x1, …, xT)<br>

#

Encoder: 입력 sequence x의 각 기호를 순차적으로 읽는 RNN<br>

각 기호를 읽을 때 RNN의 hidden state는 변경<br>

마지막 RNN의 hidden state는 전체 입력 sequence의 요약 c(=context vector)<br>

#

Decoder: hidden state h(t)가 주어지면 다음 symbol yt를 예측하여 출력 sequence를 생성하도록 훈련된 또 다른 RNN<br>

(h(t) , yt는 y(t-1)와 입력 sequence의 요약 c에 대해 조건 지정)<br>

t에서 decoder의 hidden state<br>

ht=f(h(t-1), yt-1, c)<br>

#

조건부 분포의 다음 symbol<br>

P(yt│yt-1, yt-2, …, y1, c)=g(h(t), yt-1, c)<br>

#

Encoder와 Decoder는 조건부 log-likelihood 최대화 하도록 공동 학습<br>

![4-1](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_4-1.PNG?raw=true)<br>

θ: 모델 매개변수의 집합<br>

(xn, yn): (입력 sequence, 출력 sequence)<br>

#

앞의 내용을 그림으로 표현<br>

![5-1](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_5-1.png?raw=true)<br>

Encoder의 각 step의 hidden state<br>

h(t)=f(h(t-1), xt)<br>

#

Decoder의 각 step의 hidden state<br>

ht=f(h(t-1), yt-1, c)<br>

#

source sentence가 나왔을 때 output sentence가 나올 확률 최대화<br>

![4-1](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_4-1.PNG?raw=true)<br>

#

훈련된 모델의 사용법<br>

input sequence가 주어졌을 때 target sequence 생성<br>

input sequence와 target sequence 쌍의 점수 매기기<br>

#

## 2.3 Hidden Unit that Adaptively Remembers and Forgets<br> 

계산과 구현에 더 단순한 은닉 단위 제안(GRU)<br>

(hidden activation function)<br>

![7-1](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_7-1.png?raw=true)<br>

r(reset gate): 이전 hidden state를 무시할지 여부 결정<br>

LSTM과 같이 sigmoid 함수를 통해 0~1 사이값<br>

0에 가까워지면 hidden state는 이전 hidden state를 무시하고 현재 입력으로 리셋<br>

![7(5)](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_7(5).png?raw=true)<br>

z(update gate): hidden state h를 새로운 hidden state h ̃로 업데이트 여부 선택<br>

이전 hidden state에서 현재 hidden state로 얼마나 많은 정보 전달할지 제어<br>

![7(6)](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_7(6).png?raw=true)<br>

별도의 output gate가 없음<br>

Long-term dependency 문제 극복<br>

---

# 3. Statistical Machine Translation
 

논문은 해당 모델을 SMT 시스템에 적용<br>

일반적으로 SMT에서 목표는 문장이 주어지면 아래 식을 maximize 하는 translation f를 구함<br>

![8-2](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_8-2.PNG?raw=true)<br>

p(e│f): translation model<br>

p(f): language model<br>

#

실제 SMT는 feature와 weight가 있는 log-linear model로 계산<br>

![8-1](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_8-1.PNG?raw=true)<br>

fn: n번째 특성<br>

wn: n번째 가중치<br>

Z(e): 정규화 상수<br>

---

# 4. Experiments<br>
 

Data: WMT-14<br>

영어/불어 번역 작업으로 모델 학습, 평가<br>

#

## 4.1 Data and Baseline System<br>
 

Baseline model: Moses 오픈 소스 기계 번역 모델<br>

RNN, CSLM(target language model), WP(word penalty)를 추가로 적용하여 실험<br>

#

### 4.1.1 RNN Encoder-Decoder<br>
 

1. Baseline<br>

2. RNN<br>

3. CSLM+RNN<br>

4. Baseline+CSLM+RNN+WP<br>

![9-1](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_9-1.png?raw=true)

정량평가<br>

성능이 향상된 것을 보임<br>

CSLM+RNN이 가장 높은 성능을 보임<br>

이는 CSLM과 RNN이 독립적으로 번역 시스템의 성능 향상에 기여했다고 판단<br>

![9-2](https://github.com/ornni/DL_algorithm/blob/main/Seq2Seq/image/Learning_Phrase_Representation_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation_9-2.png?raw=true)

정성평가<br>

모델이 학습한 phrase representation<br>

해당 모델이 의미적, 문법적으로 더 잘 표현<br>
