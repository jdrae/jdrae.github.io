---
title: "Self-Attention​ Encoding and Pooling​ For Speaker Recognition​"
date: 2021-08-17
categories: [Machine Learning, Paper Review]
tags: [paper review, self attention, speaker recognition]
---

a paper review based on [Self-attention encoding and pooling for speaker recognition](https://arxiv.org/abs/2008.01077)

개요
--

발성에 있어서 모든 프레임이 중요한 것은 아니고, 특정 프레임이 화자를 구분하는데 더 중요하기도 합니다. Attention 이라는 기법은 어떤 프레임에 더 집중할 수 있는지 가중치를 둠으로써 좀 더 정확한 결과가 나오게끔 합니다.

이 논문은 Self-Attention, 그 중에서도 구글이 제안한 Transformer 모델을 사용해 화자 인식을 하는 방법에 대해 소개합니다. 그리고 기존에 통계적으로 진행되었던 statistical pooling 을 벗어나, attention 을 적용한 pooling layer 를 설계해 attention 의 장점을 최대화 합니다.

화자 인식에 있어서 attention 은 주로 pooling layer 에 관해 연구되었는데, 모델로는 RNN 을 쓰거나 multi-head 방법을 사용했기 때문에 계산 과정이 많다는 단점이 있었습니다. 논문에서는 모바일 기기에서 사용할 수 있도록 파라미터 개수를 낮추는 데에 초점을 맞추었습니다. RNN 대신 attention 함수만으로 attention 인코더를 구현한 Transformer 모델은 계산 과정의 복잡도를 낮췄고, 논문의 목적에 적합한 모델이었습니다. 논문은 Transformer 를 참고해, 화자 임베딩을 추출할 때에 single-head self-attention 을 사용하였고, pooling layer 에서도 self-attention 함수를 사용해, 성능은 유지하면서 파라미터 수를 현저히 낮추었습니다. 모바일 기기에 딥러닝 화자 인증을 적용하기 위한 시도는 당시에 하나 밖에 없었기에, 유의미한 연구 결과로 보입니다.

그렇다면 attention 은 어떤 과정을 거쳐 제안 되었을까요? 또 기존의 pooling 과 논문이 제안한 attention pooling 은 어떻게 다를까요? 논문의 내용을 설명하기 전에, 간략하게 이전 연구들을 살펴보겠습니다.

Transformer 에 오기까지…
-------------------

이 부분은 화자 인식에 관한 내용이 아니라 텍스트를 도메인으로 하는 모델에 대한 설명입니다. 그러나 시간 순으로 진행되는 발성의 프레임은, 문장 속 순서가 있는 단어와 대응되며, 결국 화자 인식에도 통용될 수 있습니다. 모델은 입력 언어를 출력 언어로 바꾸는 번역 태스크 기준으로 설명을 할 것입니다. 번역은 입력 문장의 어떤 단어가 출력 문장의 단어로 전환되어야 하는 지를 판단하는데, 이는 화자 인식에서 어떤 프레임을 추출해야 화자의 특성이 잘 드러나는 지를 고민하는 것과 유사합니다.

### Seq2Seq

Attention 을 사용하는 방법은 텍스트 기반의 도메인에서 먼저 제안 된 기법입니다. 번역, 챗봇 등 다양한 길이의 문장이 주어져 또 다른 문장을 생성할 때, 일정한 모델을 적용해서 결과값을 예측해야했죠. **Sequence-to-Sequence** 라는 모델은 그러한 요구사항에 걸맞는 모델이었습니다. 재귀적인 모델인 RNN 을 사용하여 다음에 올 단어를 이전에 예측한 단어를 바탕으로 예측하기 때문에, 입력과 출력의 길이가 달라도 됩니다. 또한 입력에도 다양한 길이의 문장을 사용해도 되는데, 이는 입력 문장을 받아 context 라는 고정된 길이의 벡터(인코더의 마지막 은닉상태)를 생성하여 사용하기 때문입니다. 그러나 입력 문장을 하나의 벡터로 압축하는 것은 그만큼 정보의 손실이 있기 마련이었고, 앞 단어의 정보만을 가지고 다음 단어를 예측하는 것은 길이가 길어질 경우 정확도가 떨어지는 문제가 있었습니다(long term dependency).

### Attention Mechanism

이러한 seq2seq 모델의 문제점을 개선한 것이 **Attention Mechanism** 입니다. Attention 의 context vector 는 seq2seq 처럼 고정된 정보가 아니라, 각 단어를 예측하는 시점마다 변하는 attention score 입니다. 즉, t 번째 출력 단어를 구하기 위해 모든 입력 단어들의 은닉 상태들을 사용하여 softmax 결과를 구하고, 타겟 단어에 대한 각각의 가중치들을 이용해 출력 단어를 구하게 됩니다. 단순히 가중치 값이 높은 타겟이 결과 단어가 되지는 않고, 이 t 시점에서 구한 attention score 가 다시 t 시점의 단어를 예측하기 위한 입력으로써 작동하게 됩니다. 각 출력 단어를 예측할 때마다 전체 입력 단어를 선별적으로 고려하기 때문에, 길이가 길어져도 일관된 성능을 보일 수 있었고, 결론적으로 seq2seq 의 문제점을 개선한 모델이 되었습니다.

### Transformer

그러나 attention mechanism 은 여전히 seq2seq 의 재귀적인 방법을 따르고 있었습니다. 이에 구글은 **Transformer** 라는 모델을 제안합니다. 앞서 살펴본 seq2seq 와 attention 모델 모두 입력 단어를 처리하는 인코더, 출력 단어를 처리하는 디코더로 이루어져있습니다. Tranformer 는 이러한 인코더-디코더의 모델 구조를 가지지만, 재귀적으로 단어를 처리했던 부분을 없애고 오로지 attention 으로만 인코더와 디코더를 작성합니다. 이렇게 함으로써 계산 시간을 줄이고, 순서에 상관없이 병렬적으로 처리할 수 있게 됩니다.

> Self-attention 과 attention 의 차이는 attention 함수로 넘겨지는 값(Q,K,V)이 동일한 출처(인코더에서만, 혹은 디코더에서만)인지 아니면 다른 출처를 갖는지에 따라 있습니다. Transformer 모델의 인코더는 self-attention 이며, 디코더 레이어중 하나가 일반적인 attention 을 사용합니다.

> 순차적인 계산(sequential computation)을 줄이기 위한 방법은 많았지만 멀리 떨어진 단어들 간의 의존성 정보를 포함하기 위해 많은 계산이 필요했었습니다. Transformer 에서는 Positional Encoding 이라는 단어의 순서를 고려하면서도 계산 과정을 단순화한 기법을 사용합니다. 리뷰된 논문에서는 사용되지 않았습니다.

Pooling Layer 의 변화
------------------

화자 인식에 쓰이는 발성 데이터들은 각각 길이가 다릅니다. 그렇기에 프레임마다 벡터를 구하고, 다시 발성(utterance) 수준에서의 벡터를 얻기 위한 pooling 기법이 필요합니다. 이전에는 각 프레임의 벡터를 더하고 다시 평균을 낸 **average pooling** 기법을 사용했습니다. 이후에 프레임 벡터의 평균 뿐만 아니라 표준편차까지 고려한 **statistic pooling** 이 발표되었지만, 표준편차의 정확한 효과는 보고되지 않았다고 합니다. [참고](https://arxiv.org/abs/1803.10963)

이후에 attention 을 적용한 **attentive statistic pooling** 이 발표되었고 성능 향상도 있었지만, 논문에서는 statistical 한 부분을 제거한 **self-attention pooling** 을 제안합니다. 전자의 기법은 프레임 벡터로부터 추출된 attention score 를 가중치로 매겨 평균과 표준편차를 구하는 pooling 입니다. 한편 논문에서는 학습 가능한 파라미터를 두어 attention function 을 적용했기에, 매 학습마다 pooling 의 파라미터가 조정되는 데에 의의를 둘 수 있습니다.

모델의 구조
------

### Self-Attention Encoder

논문에서는 Transformer 의 인코더 부분을 차용하여 모델을 설계합니다. 화자 인식에서의 인코더란, 입력된 프레임의 가중치 값(attention score)을 구하고, 가중치를 다시 입력에 적용해서 화자 임베딩을 추출하기 위함입니다.

인코더는 N 개의 동일한 인코더 레이어가 쌓인 모습입니다. 각 인코더 레이어의 내부에는 self-attention mechanism 과 position-wise feed-forward 레이어가 있고, 두 레이어의 출력은 residual connection 과 layer normalization 을 거쳐 다음 층으로 전달됩니다. Transformer 는 병렬 처리를 위해 multi-head attention 을 사용하지만, 논문에서는 single-head, 하나의 헤드만 사용하여 벡터를 추출합니다.

    # class Encoder
    
    self.layer_stack = nn.ModuleList([
    	EncoderLayer(d_m, d_ff, d_k, d_v, dropout=dropout)
    	for _ in  range(n_layers)
    ])
    

인코더는 N(=2)개의 레이어로 이루어지고, 각각의 레이어는 아래와 같은 두 개의 레이어를 가집니다.

    # class EncoderLayer
    
    self.slf_attn = SelfAttention(d_m, d_k, d_v, dropout=dropout)
    self.pos_ffn = PositionwiseFeedForward(d_m, d_ff, dropout=dropout)
    

#### 1\. Single-head Self-Attention Mechanism

    # class SelfAttention
    
    self.w_q = nn.Linear(d_m, d_k)
    self.w_k = nn.Linear(d_m, d_k)
    self.w_v = nn.Linear(d_m, d_v)
    

(d\_m, d\_k) 의 차원을 가지는 학습 가능한 파라미터 w\_q 와 w\_k, 그리고 (d\_m, d\_v) 의 차원인 w\_v 를 설정합니다. 논문에서 d\_k=d\_v 를 사용합니다. 또한 기존 multi-head 에 사용되었던 식 d\_m / num\_head = d\_k = d\_v 로부터 d\_m 도 유추할 수 있습니다. single-head 이기 때문에 d\_m / 1 = d\_m = d\_k = d\_v 를 사용합니다.

    # class SelfAttention
    
    q = self.w_q(x)
    k = self.w_k(x)
    v = self.w_v(x)
    
    attn = self.attention_func(q, k, v) # scaled dot-product attention
    

(T, d\_m) 이었던 x 는 각각의 파라미터와 곱해져서 q: (T, d\_k), k: (T, d\_k), v: (T, d\_v) 로 유도됩니다. 이렇게 생성한 q,k,v 는 attention 함수의 입력으로 쓰입니다. (self-attention 기법)

    class  ScaledDotProductAttention(nn.Module):
    
    def  __init__(self, temperature, attn_dropout=0.1):
    	super().__init__()
    	self.temperature = temperature # temperature=np.power(d_k, 0.5)
    	self.softmax = nn.Softmax(dim=2)
    def  forward(self, q, k, v):
    	attn = torch.bmm(q, k.transpose(1, 2)) 
    	attn = attn / self.temperature
    	attn = self.softmax(attn)
    	attn = torch.bmm(attn, v)
    	return attn
    

이때 사용되는 attention 함수는 Transformer 논문에서 제안한 scaled dot product attention 함수입니다. 연산이 빠르기 때문에 additive attention 대신에 사용되었습니다.  
q: (T, d\_k) 와 k.transpose: (d\_k, T) 를 곱하고 softmax 함수 이후에 v: (T, d\_v) 를 곱해준 결과로 (T, d\_v) 차원의 결과가 나옵니다. v 를 다시 곱해줌으로써 특정한 프레임을 더 강조하게 됩니다.

    attn = self.layer_norm(attn + residual) # residual connection
    

attention 의 결과는 residual connection 과 layer\_norm 을 거치고 다음 레이어로 전달됩니다.

#### 2\. Position-Wise Feed-Forward

    class  PositionwiseFeedForward(nn.Module):
    
    """Implements position-wise feedforward sublayer.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    """
    
    def  __init__(self, d_m, d_ff, dropout=0.1):
    	super().__init__()
    	self.w_1 = nn.Linear(d_m, d_ff)
    	self.w_2 = nn.Linear(d_ff, d_m)
    	self.dropout = nn.Dropout(dropout)
    	self.layer_norm = nn.LayerNorm(d_m)
    
    def  forward(self, x):
    	residual = x
    	output = self.w_2(F.relu(self.w_1(x)))
    	output = self.dropout(output)
    	output = self.layer_norm(output + residual) # residual connection
    	return output
    

다음 레이어는 Linear - ReLU - Linear 로 이루어져 있습니다. 앞서 (T, d\_v) 의 결과를 (d\_m, d\_ff) 과 곱하고, 다시 (d\_ff, d\_m) 과 곱하여 (T, d\_m) 차원의 결과를 얻습니다.

### Self-Attention Pooling layer

pooling 에서는 (T, d\_m) 의 결과를 (1,d\_m) 으로 만듭니다. w\_c: (1, d\_m) 과 결과의 전치행렬인 (d\_m, T) 를 곱하고, Softmax 를 통과시켜 다시 인코더의 결과인 (T, d\_m) 과 곱합니다. 이를 거치면 (1, d\_m) 이 되어 발성 벡터를 구할 수 있습니다.

    class  SelfAttentionPooling(nn.Module):
    	def  __init__(self, d_m, dropout=0.1):
    		super().__init__()
    		self.d_m = d_m
    		self.softmax = nn.Softmax(dim=2)
    		self.w_c = nn.Linear(d_m, 1)
    
    	def  forward(self, x): # (bs, T, d_m)
    		attn = self.w_c(x).transpose(1,2) # (bs, 1, T)
    		attn = self.softmax(attn)
    		attn = torch.bmm(attn, x) # (bs, 1, d_m)
    		return attn
    

### DNN Classifier

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
    

화자 임베딩을 추출하기 위해 pooling layer 의 (1, d\_m) 은 세 개의 fully connected layers (Linear) 를 거치고 반환됩니다. 학습 단계 이후에서 화자 임베딩을 구할 때는 두 번째 레이어에서 가져옵니다.

Experiments
-----------

### Protocol

1.  Vox1
    *   train: VoxCeleb1 development set
    *   test: VoxCeleb1 test set
2.  Vox2
    *   train: VoxCeleb2 development set
    *   test: VoxCeleb1 test set
3.  Vox1-E
    *   train: Voxceleb2 development set
    *   test: Voxceleb1 development + test

### Preprocessing

1.  MFCC - 30 dimensional
2.  no data or test-time augmentation
3.  Cepstral Mean Variance Normalization
4.  train with 300 frames

### Training

1.  ReLU
2.  Adam optimizer
3.  learning rate 1e-4
4.  non linearity -> batch normalization -> TDNN
5.  PLDA backend
6.  baseline: x-vector

### Parameters

1.  N=2 encoder layers
2.  d\_k = d\_v = 512
3.  d\_ff = 2048
4.  dropout
    *   encoder: 0.1
    *   other: 0.2
5.  dimension of dense layer:
    *   first: 90
    *   others: 400 (similar like i-vector)
6.  AMSoftmax
    *   scaling factor: 30
    *   margin: 0.4

Result
------

1.  Vox1 Protocol
    *   small improvement compare to x-vector with LDA/PLDA and VGG-M
    *   using AMSoftmax increases to 8.93%(x-vector LDA/PLDA) and 7.99% (VGG-M)
2.  Vox2 Protocol / Vox1-E Protocol
    *   20% / 15% improvement compared to x-vector with LDA/PLDA
    *   ResNet-34, ResNet-50 have better results due to the much number of parameters
    *   In Vox2 SAEP performs similar with ResNet-34 with 94% less parameters

*   Impact of the key and value dimensions(d\_k = d\_v)
*   Each 64, 128, 512 dimensions have 0.83M, 0.88M, 1.16M parameters
*   dff = 1024 and dv = dk = 64​ 7.83% EER on Vox2 protocol with only 0.45M parameters​ Compared to x-vector, ​ almost one-tenth of the parameters required for the x-vector.​