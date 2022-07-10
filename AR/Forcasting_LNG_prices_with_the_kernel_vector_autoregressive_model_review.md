![title](https://github.com/ornni/DL_algorithm/blob/main/AR/image/Forcasting_LNG_prices_with_the_kernel_vector_autoregressive_model_review_title.png?raw=true)

**목차**

0. Abstract<br>

1. Instruction<br>

2. Kernel VAR<br>

2.1 VAR<br>

2.2 LS-SVM: Kernel VAR<br>

3. Forecasting Results<br>

4. Conclusions<br>

---

# 0. Abstract<br>


LNG 가격은 다변수 시게열과 관련<br>

이유: 비슷한 계약으로 거래됨<br>

#

다변수 시계열 데이터 분석을 위해, vector autoregressive model은 사용하기에 성공적인 tool 중 하나<br>

#

문제: vector autoregressive model 은 현재와 이전 데이터의 선형 회귀를 가정<br>

→ 신뢰할 수 없는 결과 발생<br>

해결: vector autoregressive model에 최소 제곱 SVM을 가중치로 적용<br>

#

다른 모델과의 비교는 제안된 kernel vector autoregressive model이 다변수 시계열 데이터에 더 잘 만족함을 보임<br>

---

# 1. Instruction

LNG 가격(물가)은 지역 수입 가격의 가중 평균으로 결정<br>

#

LNG 수입량 80~90% 장기계약(20년 이상)의 경우 협의 계약을 통해 국제 LNG 수요공급으로 가격결정<br>

(기름값 관련 공식으로 계산)<br>

나머지 SPOT이나 단기, 국내 수요 공급 관련<br>

문제: LNG 가격 예측이 중요함에도 불구하고 가격 예측이 어려움<br>

#

Vector autoregressive(VAR) 방법론은 거시경제학 간점에서 LNG 수입가격 예측에 사용됨<br>

때때로 일변량 autoregressive model에서 우수한 예측을 제공<br>

#

Support vector machine(SVM)은 현실세계에서 분류와 회귀에 성공적으로 적용<br>

LS-SVM은 매우 매력적이고 유망한 방법임은 이미 증명됨<br>

장점: 선형등식을 사용하여 계산시간을 줄이고, 해결이 간단해서 일반화된 교차 검증 함수를 사용하여 모델 선택을 쉽게 함<br>

#

**기여**

Kernel VAR을 이용하여 새로운 방법으로 LNG 가격을 에측하여 예측력을 더 정확히 함<br>

즉, 제안된 모델이 더 잘 맞고, AR(p)와 VAR(p) 모델보다 한발 에측능력이 앞선다는 뜻<br>

#

비선형 kernel VAR모델로 kernel VAR이 LNG 시장을 에측하는데 더 효과적<br>

이유: kernel VAR은 error 거리를 줄이는 것과 예측 정확도와 MSE, MSPE를 사용하여 LNG 수입 가격 예측력을 강화시키는 것을 동시에 함<br>

#

Section2: kernel VAR 제안<br>

Section3: 실제 LNG 가격 데이터를 kernel VAR을 사용하여 예측분석<br>

Section4: 결론<br>

---

# 2. Kernel VAR<br>

## 2.1 VAR<br>

![var](https://github.com/ornni/DL_algorithm/blob/main/AR/image/Forcasting%20LNG%20prices%20with%20the%20kernel%20vector%20autoregressive%20model_review_VAR.png?raw=true)

#

## 2.2 LS-SVM: Kernel VAR<br>

![LS_SVM](https://github.com/ornni/DL_algorithm/blob/main/AR/image/Forcasting%20LNG%20prices%20with%20the%20kernel%20vector%20autoregressive%20model_review_LS-SVM.png?raw=true)

---

# 3. Forecasting Results<br>


Data<br>

LNG 월별 가격<br>

2006.Sep~2015.Sep<br>

Japan, Taiwan, South Korea, China<br>

#

n=109<br>

85 training data<br>


![6-1](https://github.com/ornni/DL_algorithm/blob/main/AR/image/Forcasting%20LNG%20prices%20with%20the%20kernel%20vector%20autoregressive%20model_review_6-1.png?raw=true)

#

LNG 월별 데이터셋 그래프<br>

![7-1](https://github.com/ornni/DL_algorithm/blob/main/AR/image/Forcasting%20LNG%20prices%20with%20the%20kernel%20vector%20autoregressive%20model_review_7-1.png?raw=true)

#

MSE와 MSPE를 국가별로 계산한 결과<br>

![7-2](https://github.com/ornni/DL_algorithm/blob/main/AR/image/Forcasting%20LNG%20prices%20with%20the%20kernel%20vector%20autoregressive%20model_review_7-2.png?raw=true)

RBF kernel VAR(1)이 MSE에서 모든 국가에 대해 가장 좋은 결과를 나타낸다<br>

하지만 MSPE 측면에서는 덜 효과적으로 예측한다<br>

#

Linear kernel VAR(1)이 경우<br>

타이완을 제외하고 다른 3개 국가의 MSPE가 가장 효과적인 결과를 만든다.<br>

타이완은 AR(1)에서 가장 효과적인 결과를 나타낸다.<br>

타이완의 매년 온도는 일정하기 때문에 kernel VAR이 효과적인 결과를 만들어내기 힘들다<br>

---

# 4. Conclusions<br>


결과적으로 kernel VAR 모델을 사용하는 것이 LNG 수입 가격 예측의 정확도를 강화시킨다.<br>

앞으로 여러 방면에서 사용될 만한 모델이다.<br>
