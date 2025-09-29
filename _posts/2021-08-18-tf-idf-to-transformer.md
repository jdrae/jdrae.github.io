---
title: "NLP 훑어보기: TF-IDF 부터 Transformer 까지"
date: 2021-08-18
categories: [Machine Learning, NLP]
tags: [tf-idf, lda, rnn, transformer]
---

## TF-IDF 과 BM25

TF-IDF 는 주어진 키워드의 빈도수(term frequency)를 역문서빈도(inverse document frequency)와 곱한 것이다. 역문서 빈도는 전체 문서 중 키워드를 포함하는 문서의 값을 역으로 취한 것이다. log 를 씌우는 이유는 문서의 값이 많아질수록 IDF 의 값이 차이가 많이 나기 때문에 그 규모를 줄이기 위함이다. IDF 는 해당 키워드가 흔하지 않을 수록 높아지고, TF 는 한 문서에서 해당 키워드가 많이 출현할 수록 높아진다. 문서는 각 단어의 TF-IDF로 이루어진 벡터로 표현되고, 이후 질의문이 들어오면 질의문의 TF-IDF과 코사인 유사도를 계산해서 유사한 문서를 제시한다.

![](/assets/posts/2021-08-18-tf-idf-to-transformer/tf-idf.jpg){:width="400px"}

BM25 는 TF-IDF 의 코사인 유사도를 정규화하고 평활화(smoothing)한다. 곱셈의 왼쪽 항이 IDF 이고, 오른쪽 항이 TF 를 정규화 한 것이다. TF 부분부터 보자면, f_td 는 문서 d 에서의 t 의 빈도이다. 분모의 k, b 는 상수 파라미터이며, 해당 문서 길이 l(d) 를 평균 문서 길이 avgdl 로 나눈값도 정규화에 쓰인다. IDF 부분의 N 은 전체 문서, df_t는 해당 단어를 포함하는 문서의 개수이며 0.5 를 더함으로써 분모가 0이 되는 일이 없도록 smoothing 을 한다. *cf. 라플라스 평활화*

![](/assets/posts/2021-08-18-tf-idf-to-transformer/bm25_formula.png){:width="600px"}


## 빈도에서 의미로, 차원 축소 기법

### Latent Discrimant Analysis 선형 판별 분석

차원 축소의 간단한 방법으로 1990년대에 사용되었던 선형 판별 분석 기법이 있다. 이진 분류를 하기 위해서 미리 문서들에 분류명을 붙여둔 훈련 자료가 필요하다. 한 부류 TF-IDF 벡터들의 평균 위치(무게 중심)과 다른 부류 TF-IDF 벡터 평균 위치를 계산해 두 무게 중심을 이어 직선(무게중심의 차)을 만든다. 새로운 데이터를 분류하기 위해서 앞서 계산한 직선 벡터와 데이터의 TF-IDF 내적을 구하면 직선에 TF-IDF 를 투영한 길이를 얻을 수 있고, 이는 두 부류 중 어느 쪽에 더 치우쳐 있는지 알 수 있게 된다.

### LSA: Latent Semantic Analysis 잠재 의미 분석

잠재 의미분석(LSA, Latent Semantic Analysis)은 TF-IDF 벡터를 분석해서 문서의 주제를 추출하는 알고리즘이다.  선형 판별 분석이 이진 분류의 지도 학습 방법이었다면, LSA 는 다차원의 주제로 이루어졌으며 주제를 미리 설정할 필요가 없는 비지도 학습이다. 이는 이미지와 같은 고차원 자료의 차원을 줄이기 위한 주성분 분석(PCA, Principal Component Analysis) 기법에서 차용했다. 

![](/assets/posts/2021-08-18-tf-idf-to-transformer/lsa.jpg){:width="600px"}


LSA 는 위해 특잇값 분해(SVD, Singular Value Decomposition)를 사용하여 용어-문서 행렬(또는 용어 대신 TF-IDF)로부터 주제-문서 행렬을 생성한다. SVD 는 원래의 용어-문서 행렬을 세개의 핼렬 곱으로 분해하는데, U, V는 직교 행렬(역행렬과 전치행렬이 같음)이며, S(D, 또는 시그마)는 대각행렬이다. 이때 대각 행렬 S의 대각원소를 행렬 A의 특이값(singular value)라고 한다. 이 S 의 크기가 주제의 개수이며, 크기를 줄일 경우, 절단된 SVD(Truncated SVD)라고 한다. 

### LDA: Latent Dirichlet Allocation 잠재 디리클레 할당

LDA 는 문서가 랜덤으로 단어를 선택해서 만들어졌다고 가정한다. 그리고 이렇게 생성된 문서들은 각 주제들을 조금씩 가지고 있다. `[자전거,한강,수영복,바다]`와 같은 단어들이 있어서 문서1은 `[자전거,한강]`, 문서2는 `[수영복,바다]`, 문서3은 `[한강,바다]`와 같이 생성되었다. 이때 주제가 `[바이킹,수영,여행]`과 같이 있을 때, 문서1 은 `바이킹  0.7, 수영 0.1, 여행 0.2`와 같이 주제를 가질 수 있다. 그렇다면 이를 역으로 생각해볼 수 있다. 문서1이 `[자전거,한강]`일 때, 어떤 주제로 분류되는지 알 수 있다. 

먼저 문서 집합에 존재하는 주제 k 개를 설정한다. k 개의 주제는 디리클레 분포에 따라 각각의 문서들에 분포되었다고 가정한다. 그리고 문서 별 단어를 푸아송 분포에 따라 k 개 중 하나의 주제에 할당한다. 이때 어떤 문서의 단어가 올바른 주제로 분류되기 위해서 그 단어가 다른 문서에서는 어떤 주제로 분류되었는지, 해당 문서의 다른 단어들이 어떤 주제로 분류 되었는지를 확인한다. 이를 전체 문서의 단어에 대해 반복하면 일정한 값으로 수렴한다.

## Word2Vec

LSA 가 한 문서의 의미(주제)를 파악하는 것에 가깝다면, Word2Vec 은 개별 단어의 밀집 벡터 표현을 추출한다. 한 단어의 주변 단어들을 파악함으로써, 해당 단어의 의미를 알 수 있다는 가정에서 진행된다. Skip-gram 과 CBOW 두 가지 방법으로 단어 벡터를 구한다.

우선 Skip-gram 은 `오늘 점심은 맛있는 햄버거` 라는 문장이 있고, `맛있는`을 입력으로 넣었을 때, `오늘, 점심은, 햄버거`를 출력하는 것이다. 한편 CBOW(Continuous Bag of Words)는 `오늘, 점심은, 햄버거` 를 입력했을 때 `맛있는`을 출력한다. 단어 벡터를 추출하기 위해서 각 방법의 출력은 상관이 없고, 이를 학습하는데에 사용했던 은닉층의 가중치가 필요하다. 입력이 원핫벡터이므로 해당 입력 단어에 영향을 미친 가중치가 단어 벡터가 된다.

## CNN

주로 2차원 이미지 도메인에 사용하는 CNN 은 텍스트에 적용되었을 때 1차원의 합성곱 필터를 통해 단어들 사이의 관계를 파악할 수 있다. 단어-벡터 행렬에서 수평적으로 이동하는 합성곱 필터로 입력 전체의 합성곱 연산을 수행한다. 합성곱 연산은 해당 필터 안에 있는 단어 임베딩과 필터의 가중치를 곱하고 더해서 활성화 함수(주로 ReLU)를 적용한다. 이 연산은 각 스텝마다 독립적이므로 병렬적인 수행이 가능하다. 

각각의 합성곱 필터마다 다른 출력을 만드는데, 이 출력을 다음 단계 신경망의 입력으로 보낸다. 이때 풀링(pooling)으로 출력의 차원 축소를 수행하거나 드롭아웃(dropout)을 통해 신경망의 과대적합을 방지하는 기법을 사용한다. 마지막 층에서는 각 데이터마다 하나의 값으로 표현하기 위해 활성함수를 적용한다. 이 활성함수의 값은 손실함수로 전달되어 오차를 계산하고, 역전파를 통해 필터의 가중치 값을 갱신하게 한다. 손실함수의 오차를 최소화하기 위해 optimizer(Adam, RSMProp, etc)라는 최적화 기법을 사용한다.

## RNN과 LSTM

CNN 이나 Word2Vec 은 인접한 단어들을 통해 패턴을 파악한다. 하지만 입력 텍스트에서는 멀리 떨어져있지만 인접한 의미를 공유하는 단어도 있다. 이 의미를 파악하기 위해 순환신경망(RNN, Recursive Neural Networ)은 이번 t 단계에서의 출력을 다음 t+1 단계의 입력으로 보낸다. 

![](/assets/posts/2021-08-18-tf-idf-to-transformer/rnn.png){:width="500px"}

역전파는 BPTT(BackPropagation Through Time) 라고 하는데, 마지막 단계의 출력에서 목표값과의 오차를 구한 후 이전 단계의 가중치가 기여한 정도를 파악한다. 이때, 갱신은 가장 첫 단계에 와서 이루어진다. RNN 의 문제점은 기울기 소실 문제 또는 기울기 폭발 문제를 발생시킨다는 점이다. 신경망의 층이 깊어질 수록 역전파의 기울기가 소멸하거나 증폭되기 때문이다. 

LSTM(Long Short-Term Memory)는 기울기 문제를 완화하면서, RNN 의 기능을 강화한다. 신경망의 각 층에 상태 state 를 도입해서 다음 단계로 갈수록 입력 텍스트 전체를 아우르는 기억을 생성한다. 

![](/assets/posts/2021-08-18-tf-idf-to-transformer/lstm.png){:width="550px"}


이 기억 상태는 3개의 게이트를 통과한다. 2개의 게이트는 마스크 mask를 갖고 기억 상태를 갱신 한다. 망각게이트는 필요 없는 기억을 제거하고, 후보 게이트는 강화할 성분들을 선택한다. 마지막으로 출력 게이트는 최종적으로 갱신되 기억 벡터와 입력 데이터를 곱하고 활성화 함수를 적용해서 출력을 한다. 이 출력은 다음 단계의 LSTM 으로 내보낸다. *cf.GRU*, Gated Recurrent Unit

## Seq2Seq 과 Attention

Seq2Seq 은 LSTM(또는 GRU)으로 이루어진 인코더와 디코더 구조를 뜻한다. 입력 텍스트를 인코더에 넣어 벡터를 생성하고, 이 생성된 벡터와 기대 출력값을 디코더에 넣어 출력을 한다. 이는 입력과 출력의 길이가 다른 번역에 쓰기 적합한 형태이고, 또한 LSTM 의 특성상 가변 길이의 텍스트를 생성할 수 있다.

![](/assets/posts/2021-08-18-tf-idf-to-transformer/seq2seq.png){:width="550px"}

하지만 이런 Seq2Seq 모델은 입력 텍스트를 고정된 크기의 벡터로 표현하는데, 만약 텍스트 길이가 길 경우 벡터로 압축할 때에 텍스트의 의미를 잘 전달하지 못할 것이다. 

Attention 은 디코더의 출력 단어를 예측하는 시점에 연관이 있는 입력 단어에 집중해서 전체 입력을 다시 살펴본다. 즉 y_i 를 선택할 때 인코더 출력의 h_j를 a_ij 만큼 이용한다. y_i 의 벡터 c_i 는 ∑a_ij*h_j 가 된다.  attention score 는 현재 디코더 출력과 인코더 은닉 상태를 통해 계산하고, softmax 함수를 통과시켜 벡터를 생성한다. 이 벡터와 현재 디코더 출력은 다음 디코더 히든 상태를 출력한다.

![](/assets/posts/2021-08-18-tf-idf-to-transformer/attention.png){:width="450px"}



## Transformer

![](/assets/posts/2021-08-18-tf-idf-to-transformer/transformer.png){:width="350px"}

트랜스포머는 seq2seq의 인코더, 디코더에서 사용했던 RNN기반의 신경망 대신 attention 만으로 인코더와 디코더를 구현한 것이다. 하지만 RNN의 장점이었던 단어의 순차적인 위치 정보가 없어지는데, 이는 Positional Encoding 을 통해 해결한다. Positional Encoding 은 단어 임베딩 백터의 짝수 위치에는 사인 함수를, 홀수 위치에는 코사인 함수를 사용한 수식을 적용한다.  

이러한 단어 임베딩으로 우선 인코더에서 Self Attention 을 병렬적으로(Multi-head) 진행한다.  Attention 은 특정 단어(query)와 다른 단어(key,value)의 관계를 계산하는데, 먼저 qurey 와 전체 key 행렬을 내적해서 attention score 를 구하고 softmax 확률 값으로 만든다. 그리고 이 확률 벡터를 다시 value 와 곱하면 q,k의 관계가 v 에 가중된 결과를 얻을 수 있다. 

![](/assets/posts/2021-08-18-tf-idf-to-transformer/multiattention.png){:width="500px"}


인코더에서는 q,k,v 가 모두 같은 self attention 을 진행한다. 

Attention 을 진행한 후에 Feed Forward Network 를 거치는데, 이는 첫번째 선형 레이어에 활성화 함수 ReLU 를 적용한 값을 다시 두번째 레이어로 계산한 것이다. 이때 선형 레이어의 가중치는 한 인코더 층 내에서만 같고, 다른 층에서는 다른 값을 가진다. 

Attention 과 FFN 사이에 있는 Add&Norm 은 잔차 연결residual connection 과 층 정규화 layer normalization 를 의미한다. 잔차연결은 해당 함수에서의 입력과 출력을 더하는 것이다.

이렇게 인코더 결과물은 디코더로 전해진다. 디코더에서도 우선 self attention 을 진행하는데, 이때 mask 는 현재 시점보다 더 이후의 목표 단어를 참고해서 예측을 하지 않게끔 미래의 단어에 매우 작은 값을 입히는 기법이다. 

디코더의 두번째 attention 은 인코더의 출력값을 query로 가지며 인코더의 정보를 학습하고, 동일한 과정을 걸쳐 출력 결과를 만든다. 



**references**

* 파이썬으로 배우는 자연어 처리 인 액션, 2020
* https://wikidocs.net/book/2155
* https://m.blog.naver.com/ckdgus1433/221608376139
* https://d2l.ai/chapter_recurrent-modern/lstm.html
* http://incredible.ai/nlp/2020/02/20/Sequence-To-Sequence-with-Attention/
* https://m.blog.naver.com/ckdgus1433/221608376139
