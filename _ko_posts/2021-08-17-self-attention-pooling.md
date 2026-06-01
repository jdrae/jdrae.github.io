---
title: "Self-Attention Encoding and Pooling으로 살펴보는 화자 인식"
date: 2021-08-17
permalink: /ko/2021/08/17/self-attention-pooling/
categories: [machine-learning]
tags: [paper-review, self-attention, speaker-recognition]
translation_key: self-attention-pooling
---

이 글은 논문 [Self-attention encoding and pooling for speaker recognition](https://arxiv.org/abs/2008.01077)을 바탕으로 작성한 리뷰입니다.

## 개요

발성 데이터에서 모든 프레임이 똑같이 중요한 것은 아닙니다. 어떤 프레임은 화자를 구분하는 데 더 큰 정보를 담고 있고, 어떤 프레임은 상대적으로 덜 중요할 수 있습니다. **Attention**은 이런 차이를 가중치로 반영해, 모델이 더 중요한 프레임에 집중하도록 돕는 기법입니다.

이 논문은 Self-Attention, 그중에서도 Google이 제안한 Transformer 구조를 활용해 화자 인식을 수행하는 방법을 소개합니다. 특히 기존의 **statistical pooling**에서 벗어나, attention을 적용한 pooling layer를 설계해 self-attention의 장점을 더 적극적으로 활용합니다.

화자 인식에서 attention은 주로 pooling layer를 중심으로 연구되었습니다. 다만 기존 연구는 RNN을 사용하거나 multi-head attention을 적용하는 경우가 많아 계산량이 많다는 단점이 있었습니다. 이 논문은 모바일 기기에서도 사용할 수 있도록 파라미터 수를 줄이는 데 초점을 맞춥니다.

Transformer는 RNN 대신 attention 함수만으로 인코더를 구성하기 때문에 계산 복잡도를 낮출 수 있습니다. 논문에서는 이 구조를 참고해 화자 임베딩을 추출할 때 **single-head self-attention**을 사용하고, pooling layer에도 self-attention 함수를 적용합니다. 그 결과 성능은 유지하면서 파라미터 수를 크게 줄였습니다. 모바일 기기에 딥러닝 기반 화자 인증을 적용하려는 시도가 당시 많지 않았다는 점에서도 의미 있는 연구라고 볼 수 있습니다.

그렇다면 attention은 어떤 과정을 거쳐 제안되었을까요? 그리고 기존 pooling과 논문이 제안한 attention pooling은 어떻게 다를까요? 본격적으로 논문 내용을 보기 전에, 먼저 관련 배경을 간단히 살펴보겠습니다.

## Transformer에 오기까지

이 부분은 화자 인식이 아니라 텍스트 도메인의 모델을 기준으로 설명합니다. 다만 시간 순서가 있는 발성 프레임은 문장 안에서 순서가 있는 단어와 대응해서 이해할 수 있습니다. 즉, 번역 모델에서 입력 문장의 어떤 단어에 집중할지 판단하는 문제는 화자 인식에서 어떤 프레임이 화자 특성을 잘 드러내는지 찾는 문제와 닮아 있습니다.

### Seq2Seq

Attention은 텍스트 기반 도메인에서 먼저 제안된 기법입니다. 번역이나 챗봇처럼 길이가 다양한 문장을 입력받아 또 다른 문장을 생성해야 할 때, 입력과 출력의 길이가 달라도 처리할 수 있는 모델이 필요했습니다. **Sequence-to-Sequence(Seq2Seq)**는 이런 요구에 잘 맞는 구조였습니다.

Seq2Seq는 RNN을 사용해 이전에 예측한 단어를 바탕으로 다음 단어를 예측합니다. 입력 문장의 길이가 달라도 사용할 수 있는데, 입력 문장을 인코더의 마지막 은닉 상태인 고정 길이의 **context vector**로 압축하기 때문입니다.

하지만 입력 문장을 하나의 벡터로 압축하면 정보 손실이 발생할 수밖에 없습니다. 또한 앞쪽 단어의 정보만으로 뒤쪽 단어를 예측하다 보니, 문장이 길어질수록 성능이 떨어지는 **long-term dependency** 문제가 있었습니다.

### Attention Mechanism

이런 Seq2Seq의 문제를 개선한 것이 **Attention Mechanism**입니다.

Attention에서 context vector는 고정된 하나의 정보가 아니라, 각 출력 단어를 예측하는 시점마다 달라지는 attention score를 바탕으로 계산됩니다. 예를 들어 `t`번째 출력 단어를 구할 때, 모델은 모든 입력 단어의 은닉 상태를 참고해 softmax 결과를 구하고, 각 입력 단어에 대한 가중치를 사용해 현재 시점의 정보를 만듭니다.

단순히 가중치가 가장 높은 단어가 곧바로 출력 단어가 되는 것은 아닙니다. 해당 시점에서 계산한 attention score가 다시 `t`번째 단어를 예측하기 위한 입력으로 작동합니다. 이렇게 각 출력 단어를 예측할 때마다 전체 입력 문장을 선별적으로 고려하기 때문에, 문장이 길어져도 더 안정적인 성능을 기대할 수 있습니다.

### Transformer

하지만 Attention Mechanism도 여전히 Seq2Seq의 재귀적인 구조를 따르고 있었습니다. 이에 Google은 **Transformer**를 제안합니다.

Seq2Seq와 Attention 기반 모델은 모두 입력 단어를 처리하는 인코더와 출력 단어를 처리하는 디코더로 구성됩니다. Transformer도 인코더-디코더 구조를 사용하지만, 단어를 재귀적으로 처리하는 방식을 제거하고 attention만으로 인코더와 디코더를 구성합니다. 그 결과 계산 시간을 줄이고, 입력을 병렬적으로 처리할 수 있게 됩니다.

> Self-attention과 일반 attention의 차이는 attention 함수에 전달되는 `Q`, `K`, `V`가 같은 출처에서 오는지, 서로 다른 출처에서 오는지에 있습니다. Transformer의 인코더는 self-attention을 사용하고, 디코더의 일부 레이어는 일반 attention을 사용합니다.

> 순차 계산을 줄이기 위한 방법은 이전에도 있었지만, 멀리 떨어진 단어 사이의 의존성을 반영하려면 많은 계산이 필요했습니다. Transformer는 **Positional Encoding**을 사용해 단어 순서를 반영하면서도 계산 과정을 단순화합니다. 다만 리뷰 대상 논문에서는 positional encoding을 사용하지 않습니다.

## Pooling Layer의 변화

화자 인식에 사용하는 발성 데이터는 길이가 제각각입니다. 따라서 프레임마다 벡터를 구한 뒤, 이를 다시 발성(utterance) 수준의 벡터로 변환하는 pooling 기법이 필요합니다.

초기에는 각 프레임 벡터를 더하고 평균을 내는 **average pooling**을 사용했습니다. 이후에는 프레임 벡터의 평균뿐 아니라 표준편차까지 함께 고려하는 **statistic pooling**이 제안되었습니다. 다만 논문에 따르면, 표준편차가 실제로 어떤 효과를 주는지는 명확히 보고되지 않았다고 합니다. 관련 내용은 [Attentive Statistics Pooling for Deep Speaker Embedding](https://arxiv.org/abs/1803.10963)에서 확인할 수 있습니다.

이후 attention을 적용한 **attentive statistic pooling**이 발표되었고 성능 향상도 있었습니다. 반면 이 논문은 statistical한 부분을 제거한 **self-attention pooling**을 제안합니다.

Attentive statistic pooling은 프레임 벡터에서 추출한 attention score를 가중치로 사용해 평균과 표준편차를 구합니다. 반면 이 논문은 학습 가능한 파라미터를 두고 attention function을 적용합니다. 따라서 학습이 진행될 때마다 pooling layer의 파라미터도 함께 조정된다는 점에 의미가 있습니다.

## 모델 구조

### Self-Attention Encoder

논문에서는 Transformer의 인코더 부분을 차용해 모델을 설계합니다. 화자 인식에서 인코더의 역할은 입력 프레임의 attention score를 구하고, 이 가중치를 다시 입력에 적용해 화자 임베딩을 추출하는 것입니다.

인코더는 `N`개의 동일한 인코더 레이어를 쌓은 구조입니다. 각 인코더 레이어 안에는 self-attention mechanism과 position-wise feed-forward layer가 있습니다. 두 레이어의 출력은 residual connection과 layer normalization을 거쳐 다음 층으로 전달됩니다.

Transformer는 병렬 처리를 위해 multi-head attention을 사용하지만, 이 논문에서는 파라미터 수를 줄이기 위해 하나의 헤드만 사용하는 **single-head attention**을 적용합니다.

```python
# class Encoder

self.layer_stack = nn.ModuleList([
    EncoderLayer(d_m, d_ff, d_k, d_v, dropout=dropout)
    for _ in range(n_layers)
])
```

인코더는 `N=2`개의 레이어로 이루어져 있고, 각각의 레이어는 아래 두 가지 레이어를 가집니다.

```python
# class EncoderLayer

self.slf_attn = SelfAttention(d_m, d_k, d_v, dropout=dropout)
self.pos_ffn = PositionwiseFeedForward(d_m, d_ff, dropout=dropout)
```

### 1. Single-Head Self-Attention Mechanism

```python
# class SelfAttention

self.w_q = nn.Linear(d_m, d_k)
self.w_k = nn.Linear(d_m, d_k)
self.w_v = nn.Linear(d_m, d_v)
```

먼저 `(d_m, d_k)` 차원의 학습 가능한 파라미터 `w_q`, `w_k`와 `(d_m, d_v)` 차원의 `w_v`를 설정합니다. 논문에서는 `d_k = d_v`를 사용합니다.

기존 multi-head attention에서는 보통 `d_m / num_head = d_k = d_v` 관계를 사용합니다. 이 논문은 single-head이므로 `d_m / 1 = d_m = d_k = d_v`로 볼 수 있습니다.

```python
# class SelfAttention

q = self.w_q(x)
k = self.w_k(x)
v = self.w_v(x)

attn = self.attention_func(q, k, v) # scaled dot-product attention
```

입력 `x`의 차원이 `(T, d_m)`이라면, 각 파라미터와 곱해진 뒤 `q: (T, d_k)`, `k: (T, d_k)`, `v: (T, d_v)`가 됩니다. 이렇게 생성한 `q`, `k`, `v`는 attention 함수의 입력으로 사용됩니다.

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature # temperature=np.power(d_k, 0.5)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = torch.bmm(attn, v)
        return attn
```

여기서 사용되는 attention 함수는 Transformer 논문에서 제안한 **scaled dot-product attention**입니다. additive attention보다 연산이 빠르기 때문에 이 방식을 사용합니다.

`q: (T, d_k)`와 `k.transpose: (d_k, T)`를 곱하고 softmax를 통과시킨 뒤, 다시 `v: (T, d_v)`를 곱합니다. 그 결과 `(T, d_v)` 차원의 출력이 만들어집니다. 마지막에 `v`를 곱하는 과정에서 특정 프레임의 정보가 더 강조됩니다.

```python
attn = self.layer_norm(attn + residual) # residual connection
```

attention 결과는 residual connection과 layer normalization을 거친 뒤 다음 레이어로 전달됩니다.

### 2. Position-Wise Feed-Forward

```python
class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_m, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_m, d_ff)
        self.w_2 = nn.Linear(d_ff, d_m)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_m)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # residual connection
        return output
```

다음 레이어는 `Linear - ReLU - Linear` 구조입니다. 앞서 얻은 `(T, d_v)` 차원의 결과를 `(d_m, d_ff)`와 곱하고, 다시 `(d_ff, d_m)`과 곱해 `(T, d_m)` 차원의 결과를 얻습니다.

## Self-Attention Pooling Layer

pooling layer에서는 `(T, d_m)` 차원의 결과를 `(1, d_m)` 차원의 발성 벡터로 변환합니다.

먼저 `w_c: (1, d_m)`와 인코더 출력의 전치행렬 `(d_m, T)`를 곱합니다. 이후 softmax를 통과시켜 attention score를 만들고, 다시 인코더 출력 `(T, d_m)`과 곱합니다. 이 과정을 거치면 최종적으로 `(1, d_m)` 차원의 발성 벡터를 얻을 수 있습니다.

```python
class SelfAttentionPooling(nn.Module):
    def __init__(self, d_m, dropout=0.1):
        super().__init__()
        self.d_m = d_m
        self.softmax = nn.Softmax(dim=2)
        self.w_c = nn.Linear(d_m, 1)

    def forward(self, x): # (bs, T, d_m)
        attn = self.w_c(x).transpose(1, 2) # (bs, 1, T)
        attn = self.softmax(attn)
        attn = torch.bmm(attn, x) # (bs, 1, d_m)
        return attn
```

## DNN Classifier

```python
# class Transformer

def forward(self, x, is_test=False):
    x = self.encoder(x)
    x = self.pooling(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    if is_test:
        return torch.squeeze(x)
    x = self.fc3(x)
    x = self.relu(x)
    return torch.squeeze(x)
```

화자 임베딩을 추출하기 위해 pooling layer의 `(1, d_m)` 출력은 세 개의 fully connected layer를 거칩니다. 학습 이후 실제 화자 임베딩을 구할 때는 두 번째 fully connected layer의 출력을 사용합니다.

## 실험 설정

### Protocol

1. **Vox1**
   - train: VoxCeleb1 development set
   - test: VoxCeleb1 test set
2. **Vox2**
   - train: VoxCeleb2 development set
   - test: VoxCeleb1 test set
3. **Vox1-E**
   - train: VoxCeleb2 development set
   - test: VoxCeleb1 development + test

### Preprocessing

1. 30-dimensional MFCC
2. Data augmentation과 test-time augmentation은 사용하지 않음
3. Cepstral Mean Variance Normalization 적용
4. 300 frames 기준으로 학습

### Training

1. ReLU
2. Adam optimizer
3. Learning rate: `1e-4`
4. Non-linearity, batch normalization, TDNN 사용
5. PLDA backend
6. Baseline: x-vector

### Parameters

1. Encoder layer 수: `N = 2`
2. `d_k = d_v = 512`
3. `d_ff = 2048`
4. Dropout
   - encoder: `0.1`
   - other: `0.2`
5. Dense layer dimension
   - first: `90`
   - others: `400` (i-vector와 유사)
6. AMSoftmax
   - scaling factor: `30`
   - margin: `0.4`

## 결과

### Vox1 Protocol

- x-vector with LDA/PLDA와 VGG-M 대비 소폭 개선되었습니다.
- AMSoftmax를 사용했을 때 x-vector LDA/PLDA 대비 `8.93%`, VGG-M 대비 `7.99%` 성능이 향상되었습니다.

### Vox2 Protocol / Vox1-E Protocol

- x-vector with LDA/PLDA 대비 약 `20%`, `15%` 개선되었습니다.
- ResNet-34와 ResNet-50은 훨씬 많은 파라미터를 사용하기 때문에 더 좋은 결과를 보였습니다.
- Vox2에서는 SAEP가 ResNet-34와 유사한 성능을 보이면서도 파라미터 수는 약 `94%` 적었습니다.

### Key와 Value 차원의 영향

- `d_k = d_v` 값을 각각 `64`, `128`, `512`로 설정했을 때 파라미터 수는 `0.83M`, `0.88M`, `1.16M`이었습니다.
- `d_ff = 1024`, `d_v = d_k = 64`일 때 Vox2 protocol에서 `7.83%` EER을 기록했고, 파라미터 수는 `0.45M`에 불과했습니다.
- 이는 x-vector와 비교했을 때 필요한 파라미터 수가 거의 10분의 1 수준이라는 점에서 의미가 있습니다.

## 정리

이 논문은 화자 인식 모델에 self-attention encoder와 self-attention pooling을 적용해, 성능을 유지하면서도 파라미터 수를 크게 줄일 수 있음을 보여줍니다. 특히 모바일 환경처럼 계산 자원이 제한된 상황에서 사용할 수 있는 화자 인증 모델을 고민했다는 점이 인상적입니다.

핵심은 모든 프레임을 동일하게 다루지 않고, 화자 정보를 더 잘 담고 있는 프레임에 attention을 주는 것입니다. 기존의 statistical pooling이 평균과 표준편차에 기반해 발성 벡터를 만들었다면, self-attention pooling은 학습 가능한 파라미터를 통해 프레임별 중요도를 직접 조정합니다.

Transformer의 아이디어가 자연어 처리에만 머무르지 않고, 화자 인식처럼 시간 순서가 있는 다른 도메인에도 응용될 수 있다는 점을 잘 보여주는 사례라고 생각합니다.
