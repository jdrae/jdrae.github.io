---
title: "Speaker Recognition Through Self-Attention Encoding and Pooling"
date: 2021-08-17
categories: [machine-learning]
tags: [paper-review, self-attention, speaker-recognition]
translation_key: self-attention-pooling
---

This post is a review based on the paper [Self-attention encoding and pooling for speaker recognition](https://arxiv.org/abs/2008.01077).

## Overview

Not every frame in utterance data is equally important. Some frames contain more information for distinguishing a speaker, while others may be relatively less important. **Attention** is a technique that reflects these differences as weights and helps the model focus on more important frames.

This paper introduces a method for performing speaker recognition using Self-Attention, specifically the Transformer architecture proposed by Google. In particular, it moves away from conventional **statistical pooling** and designs a pooling layer that applies attention, making more active use of the strengths of self-attention.

In speaker recognition, attention has mostly been studied around the pooling layer. However, many previous studies used RNNs or applied multi-head attention, which had the downside of high computational cost. This paper focuses on reducing the number of parameters so that the model can be used even on mobile devices.

Because Transformer builds its encoder using only attention functions instead of RNNs, it can reduce computational complexity. Referring to this structure, the paper uses **single-head self-attention** when extracting speaker embeddings, and also applies a self-attention function to the pooling layer. As a result, it significantly reduces the number of parameters while maintaining performance. It is also meaningful because there were not many attempts at the time to apply deep learning-based speaker authentication to mobile devices.

Then how was attention proposed, and how is the attention pooling proposed in this paper different from existing pooling methods? Before looking at the paper in detail, let's first briefly review the background.

## Getting to Transformer

This section explains the background using models from the text domain rather than speaker recognition. However, utterance frames with temporal order can be understood as analogous to words with order in a sentence. In other words, the problem of deciding which words in an input sentence to focus on in a translation model resembles the problem of finding which frames best reveal speaker characteristics in speaker recognition.

### Seq2Seq

Attention was first proposed in text-based domains. For tasks such as translation or chatbots, where a model must receive sentences of varying lengths and generate another sentence, a model was needed that could handle different input and output lengths. **Sequence-to-Sequence (Seq2Seq)** was well suited to this need.

Seq2Seq uses an RNN to predict the next word based on previously predicted words. It can handle input sentences of different lengths because it compresses the input sentence into a fixed-length **context vector**, the final hidden state of the encoder.

However, compressing the input sentence into a single vector inevitably causes information loss. Also, because later words are predicted only from information about earlier words, performance degrades as sentences become longer. This is the **long-term dependency** problem.

### Attention Mechanism

The **Attention Mechanism** improves on these problems in Seq2Seq.

In attention, the context vector is not a single fixed piece of information. Instead, it is computed based on attention scores that change at each point when an output word is predicted. For example, when predicting the `t`th output word, the model refers to the hidden states of all input words, computes a softmax result, and uses the weights for each input word to create information for the current step.

The word with the highest weight does not simply become the output word. The attention score calculated at that step acts again as input for predicting the `t`th word. Because the entire input sentence is selectively considered each time an output word is predicted, more stable performance can be expected even for longer sentences.

### Transformer

However, the Attention Mechanism still followed the recursive structure of Seq2Seq. Google then proposed **Transformer**.

Both Seq2Seq and attention-based models consist of an encoder that processes input words and a decoder that processes output words. Transformer also uses an encoder-decoder structure, but removes the recursive word-by-word processing method and builds the encoder and decoder only with attention. As a result, computation time is reduced and inputs can be processed in parallel.

> The difference between self-attention and regular attention lies in whether `Q`, `K`, and `V` passed to the attention function come from the same source or different sources. The Transformer encoder uses self-attention, while some decoder layers use regular attention.

> Methods for reducing sequential computation existed before, but reflecting dependencies between distant words required a lot of computation. Transformer uses **Positional Encoding** to reflect word order while simplifying the computation process. However, the paper reviewed here does not use positional encoding.

## Changes in the Pooling Layer

Utterance data used in speaker recognition has varying lengths. Therefore, after obtaining vectors for each frame, a pooling technique is needed to convert them into an utterance-level vector.

Early methods used **average pooling**, which sums frame vectors and takes their average. Later, **statistic pooling** was proposed, which considers not only the mean of frame vectors but also their standard deviation. According to the paper, however, it has not been clearly reported what effect the standard deviation actually provides. Related details can be found in [Attentive Statistics Pooling for Deep Speaker Embedding](https://arxiv.org/abs/1803.10963).

After that, **attentive statistic pooling**, which applies attention, was introduced and showed performance improvements. In contrast, this paper proposes **self-attention pooling**, which removes the statistical component.

Attentive statistic pooling uses attention scores extracted from frame vectors as weights to compute the mean and standard deviation. This paper, on the other hand, introduces learnable parameters and applies an attention function. The meaningful point is that the parameters of the pooling layer are adjusted together as training progresses.

## Model Architecture

### Self-Attention Encoder

The paper designs the model by borrowing the encoder part of Transformer. In speaker recognition, the encoder's role is to compute attention scores for input frames and apply these weights back to the input to extract speaker embeddings.

The encoder is a stack of `N` identical encoder layers. Each encoder layer contains a self-attention mechanism and a position-wise feed-forward layer. The outputs of both layers pass through residual connection and layer normalization before being passed to the next layer.

Transformer uses multi-head attention for parallel processing, but this paper applies **single-head attention** to reduce the number of parameters.

```python
# class Encoder

self.layer_stack = nn.ModuleList([
    EncoderLayer(d_m, d_ff, d_k, d_v, dropout=dropout)
    for _ in range(n_layers)
])
```

The encoder consists of `N=2` layers, and each layer has the following two layers.

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

First, learnable parameters `w_q` and `w_k` with dimensions `(d_m, d_k)`, and `w_v` with dimensions `(d_m, d_v)`, are defined. The paper uses `d_k = d_v`.

In conventional multi-head attention, the relationship is usually `d_m / num_head = d_k = d_v`. Since this paper uses a single head, this can be viewed as `d_m / 1 = d_m = d_k = d_v`.

```python
# class SelfAttention

q = self.w_q(x)
k = self.w_k(x)
v = self.w_v(x)

attn = self.attention_func(q, k, v) # scaled dot-product attention
```

If the input `x` has shape `(T, d_m)`, after multiplication with each parameter, the resulting tensors become `q: (T, d_k)`, `k: (T, d_k)`, and `v: (T, d_v)`. The generated `q`, `k`, and `v` are used as inputs to the attention function.

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

The attention function used here is **scaled dot-product attention**, proposed in the Transformer paper. This method is used because it is faster than additive attention.

It multiplies `q: (T, d_k)` by `k.transpose: (d_k, T)`, passes the result through softmax, and then multiplies it again by `v: (T, d_v)`. The output has shape `(T, d_v)`. In the final multiplication by `v`, information from specific frames is emphasized more strongly.

```python
attn = self.layer_norm(attn + residual) # residual connection
```

The attention result passes through residual connection and layer normalization before being passed to the next layer.

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

The next layer has a `Linear - ReLU - Linear` structure. The `(T, d_v)` result obtained earlier is multiplied by `(d_m, d_ff)`, then again by `(d_ff, d_m)`, producing a `(T, d_m)` result.

## Self-Attention Pooling Layer

In the pooling layer, the `(T, d_m)` result is converted into an utterance vector with shape `(1, d_m)`.

First, `w_c: (1, d_m)` is multiplied by the transpose of the encoder output `(d_m, T)`. The result is passed through softmax to create attention scores, then multiplied again by the encoder output `(T, d_m)`. Through this process, a final utterance vector with shape `(1, d_m)` is obtained.

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

To extract speaker embeddings, the `(1, d_m)` output of the pooling layer passes through three fully connected layers. After training, the output of the second fully connected layer is used when obtaining actual speaker embeddings.

## Experimental Setup

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
2. Data augmentation and test-time augmentation are not used
3. Cepstral Mean Variance Normalization is applied
4. Training is based on 300 frames

### Training

1. ReLU
2. Adam optimizer
3. Learning rate: `1e-4`
4. Non-linearity, batch normalization, TDNN are used
5. PLDA backend
6. Baseline: x-vector

### Parameters

1. Number of encoder layers: `N = 2`
2. `d_k = d_v = 512`
3. `d_ff = 2048`
4. Dropout
   - encoder: `0.1`
   - other: `0.2`
5. Dense layer dimension
   - first: `90`
   - others: `400` (similar to i-vector)
6. AMSoftmax
   - scaling factor: `30`
   - margin: `0.4`

## Results

### Vox1 Protocol

- It showed a slight improvement over x-vector with LDA/PLDA and VGG-M.
- When AMSoftmax was used, performance improved by `8.93%` over x-vector LDA/PLDA and `7.99%` over VGG-M.

### Vox2 Protocol / Vox1-E Protocol

- It improved by about `20%` and `15%` over x-vector with LDA/PLDA.
- ResNet-34 and ResNet-50 showed better results because they use far more parameters.
- In Vox2, SAEP showed performance similar to ResNet-34 while using about `94%` fewer parameters.

### Effect of Key and Value Dimensions

- When `d_k = d_v` was set to `64`, `128`, and `512`, the number of parameters was `0.83M`, `0.88M`, and `1.16M`, respectively.
- When `d_ff = 1024` and `d_v = d_k = 64`, it recorded `7.83%` EER on the Vox2 protocol, with only `0.45M` parameters.
- This is meaningful because it requires almost one-tenth the number of parameters compared with x-vector.

## Summary

This paper shows that applying a self-attention encoder and self-attention pooling to a speaker recognition model can significantly reduce the number of parameters while maintaining performance. I found it especially interesting that it considered a speaker authentication model usable in environments with limited computational resources, such as mobile devices.

The core idea is not to treat every frame equally, but to give attention to frames that contain more speaker information. Existing statistical pooling creates utterance vectors based on mean and standard deviation, while self-attention pooling directly adjusts frame-level importance through learnable parameters.

I think this is a good example showing that the ideas behind Transformer are not limited to natural language processing, but can also be applied to other domains with temporal order, such as speaker recognition.
