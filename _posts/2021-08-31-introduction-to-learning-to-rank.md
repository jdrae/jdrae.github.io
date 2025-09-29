---
title: "검색의 순위를 계산하는 법 - Learning to Rank"
date: 2021-08-31
categories: [Algorithm]
tags: [search algorithm]
---

웹에는 수많은 문서들이 있고, 우리는 세상에 존재하는 거의 모든 정보를 얻을 수 있게 되었습니다. 이제 중요한 것은 그 많은 정보 중에서 우리가 원하는 정보를 어떻게 얻을 수 있느냐가 문제가 되었죠. 단순히 생각한다면 내가 검색한 '플라톤'이라는 키워드를 포함하는 문서를 전부 보여달라고 할 수 있습니다. 하지만 그것이 과연 옳을까요?  모든 문서를 하나하나 읽어보면서 플라톤에 대한 정보를 찾을 바에는 차라리 철학과 교수님께 이메일을 보내는 것이 더 빠를 수도 있습니다. 
그래서 필요한 것이 검색 결과의 순위를 매기는 법입니다. 내가 검색한 내용을 포함하는 엄청나게 많은 문서 중에서 가장 잘 쓰여있고 가장 도움이 되는 문서를 순서대로 보여주는 것이죠. 이렇게 우리는 벌써 **정보 검색**(Information Retrieval)의 핵심적인 내용을 간파하게 되었습니다. 첫째, 검색어를 포함하는 문서 보여주기. 둘째, '가장 도움이 되는' 이 어떤 것인지 정의하기. 셋째, 순서를 계산하는 법을 배우기.

## 검색의 기본적인 원리

더 세부적인 내용을 다루기 전에, 검색이 어떻게 이루어지는지 과정을 살펴봅시다. *(검색엔진의 범위가 어디까지인지 확실하지 않아, 검색봇이라는 명칭으로 설명하겠습니다.)*
### 문서의 색인어 추출
검색을 시작하기 전에, 먼저 결과로 보여줄 데이터가 있어야 합니다. 크롤러는 웹 사이트의 다양한 문서를 수집하고 데이터베이스에 저장하는데, 이때 문서의 정보도 함께 분석하여 저장합니다. 문서의 정보는 바로 단어 일텐데요, 단어는 이후에 검색에 중요하게 쓰입니다. 예를 들어 '플라톤'이라는 단어가 들어간 문서를 찾기 위해서는 데이터베이스의 모든 문서를 하나하나 살펴보면서 해당 단어를 찾는 것보다는, 차라리 '플라톤'이라는 단어와 연결된 문서를 저장해 두는 것이 좋을 것입니다. 
|단어|문서  |
|--|--|
| 플라톤 | 문서1, 문서2, ... |
| 니체 | 문서2, 문서3, ... |

이렇게 단어에 문서를 연결한 것을 **역색인**(inverted index)이라고 합니다. 문서에서 색인으로 쓸 단어를 추출하기 위해서 적절한 형태소 분석과 불용어 제거도 필요합니다.

### 사용자의 질의와 의도
이제 사용자는 검색창에 자신이 알고싶은 내용을 작성합니다. '플라톤 생애', '한국 올림픽 일정', '조깅할때 좋은 신발' 등.. 이를 **질의**(query)라고 합니다. 보다 정확한 결과를 보여주기 위해서 검색봇은 **사용자의 의도**(user intent)를 파악하려고 합니다. 예를 들어 도쿄 올림픽이 진행되는 시점에서 '한국 올림픽 일정'이라고 검색했을 때, '한국에서 했던 평창 올림픽의 일정'보다는 '도쿄 올림픽의 한국 일정'에 대한 정보를 보여주는 것이 시의적절할 것입니다. 또한 음성 검색이 활발해지고 있기 때문에 단순한 키워드 위주의 검색 보다는 '플라톤이 언제 태어났어?'와 같은 자연어의 처리가 중요해졌습니다.
### DB 검색과 순위 계산
사용자의 질의와 의도를 얻으면, 검색봇은 우선 역색인을 이용해 몇가지 문서를 추려낼 것입니다. '조깅할때 좋은 신발'의 질의의 결과로 '조깅', '신발', '좋은'과 같은 단어가 포함된 문서를 전체 문서에서 가져오는 것이죠. 그 다음에 사용자의 의도, 문서의 신뢰도, 질의와 문서의 연관관계 등을 종합하여 순위를 계산하게 됩니다. 

## Hey Google, Learn to Rank.
Learning to Rank(LTR)은 Machine-Learned Ranking(MLR) 이라고도 합니다. 앞서 살펴봤듯이, 질의에 쓰인 키워드의 통계적 측면만 중요한 것이 아니라, 다양한 측면의 특징(features)들을 추출하고 계산해서 최적의 검색 결과를 **학습**하기 때문입니다. 
LTR 모델은 다음과 같은 방법으로 제작됩니다.
1. 평가 리스트(judgment list) 제작하기
	*  주어진 질의에 적절한 문서 매칭
2. 특징 정의하기
	* 클릭 횟수, 좋아요 횟수 등 모델이 학습해야하는 특징
3. 훈련 데이터 제작하기(평가 리스트 문서의 특징 설정하기)
4. 모델 훈련 및 평가하기
	* precision: 모델이 반환한 검색 결과중에 실제 관련있는 검색 결과 비율 *true positives / total results*
	* recall: 전체 관련있는 결과 중에 모델이 반환한 관련있는 결과 비율 *true positives / (true positives + false negatives)*
	* nDCG
5. 검색엔진에 적용하기

### nDCG: Nomralized Discounted Cumulative Gain
모델은 오차를 줄이는 방향으로 학습을 진행합니다. 이때 오차를 계산하는 방법으로 다양한 방법이 있지만, 대표적으로 nDCG를 살펴보겠습니다.

<img src="img/ndcg.png" width="300px"/>

DCG_p 는 상위 p 개의 검색 결과를 랭킹 순서에 따라 비중을 줄여 관련도를 계산합니다. 하지만 이는 추천 모델마다 결과 범위가 다양할 수 있기 때문에 정규화를 시켜 서로 다른 모델을 비교할 수 있게 합니다. DCG_p 의 값을 IDCG_p 로 나눠서 정규화를 시키는데, IDCG_p 는 상위 p 개 검색 결과의 이상적인 관련도를 계산한 것입니다. nDCG 는 값이 클 수록 정확한 결과임을 뜻합니다.

### 접근 방법
순서가 있는 검색 결과를 반환하기 위해 함수 f 를 정의해봅시다. f(d,q)는 문서 d 와 질의문 q 를 입력받아 문서의 순위를 반환할 것입니다. 그리고 모든 문서를 f(d,q) 에 따라 정렬했을 때, nDCG 가 최대화 되도록 함수를 작성해야 합니다. 이런 함수를 만들기 위해 다양하게 접근을 할 수 있습니다.

#### Point wise Learning to Rank
아주 간단한 공식으로 `f(d,q) = 10 * titleScore(d,q) + 2 * descScore(d,q)`[(출처)](https://opensourceconnections.com/blog/2017/08/03/search-as-machine-learning-prob/)를 생각해 볼 수 있습니다. 모든 문서의 점수를 계산하고 점수가 높은 순서대로 순위를 정렬하게 됩니다. 이 함수는 계산된 각 문서의 점수와 목표했던 점수의 차이를 계산하는 식으로 오차를 개선할 것입니다. 즉 한 번에 하나의 문서를 살펴보는 point wise 입니다. 하지만 1위 부터 100위까지 동일하게 오차를 계산한다면 1위의 중요성과 100위의 중요성이 같아집니다. 일반적으로 상위권 검색 결과의 비중을 중요하게 두는 것과는 다른 접근 방식입니다.

#### Pair wise Learning to Rank
Pair wise 는 두 문서의 쌍(pair)을 비교하면서 순위를 조정합니다. 문서 x_i 와 x_j 의 쌍 (x_i, x_j) 가 있을 때, x_i 가 x_j 보다 순위가 높으면 1, 순위가 낮으면 -1 로 값을 매깁니다. 
x_i 가 x_j 보다 순위가 높다는 의미를 생각해보면, 두 문서의  차이에 따라 분류를 할 수 있다고도 생각해볼 수 있습니다. 이러한 아이디어를 바탕으로 **RankSVM** 은 각각의 문서들을 분류하는 결정경계를 찾고 이를 통해 방향 벡터를 얻습니다.

#### List wise Learning to Rank
List wise 는 전체 문서 목록의 이상적인 순서를 모델의 결과 순서와 비교합니다. 예를 들어 1위부터 100위까지의 순서는 100! 중 1개인 순열로 볼 수 있습니다. 모델이 반환한 검색 결과 순열이 실제 목표하는 순열일 확률을 계산하는 것으로 비교를 합니다. 순열을 계산할 때에 1위가 문서i일 확률, 2위가 문서j일 확률 .. 처럼 계산되기 때문에, 상위권 순위일 수록 가중된 결과를 얻습니다. ~~(정확한 계산 과정 필요)~~ 
전체 문서의 순위 확률을 계산하기에는 계산량이 많기 때문에, **Top one probability** 로 계산하기도 합니다.

---
#### References
- https://www.google.com/search/howsearchworks/
- https://elasticsearch-learning-to-rank.readthedocs.io/en/latest/core-concepts.html
- https://opensourceconnections.com/blog/2017/02/24/what-is-learning-to-rank/
- https://opensourceconnections.com/blog/2017/08/03/search-as-machine-learning-prob/
- https://www.youtube.com/watch?v=eMuepJpjUjI&ab_channel=Lucidworks
- https://lucidworks.com/post/abcs-learning-to-rank/
- https://ride-or-die.info/normalized-discounted-cumulative-gain/

#### More to read
- 구글 서치 블로그 https://blog.google/products/search/
- 페이지랭크 http://infolab.stanford.edu/~backrub/google.html
-  크롤링과 색인 https://developers.google.com/search/docs/advanced/crawling/overview?hl=ko
- 지식 그래프 https://blog.google/products/search/introducing-knowledge-graph-things-not/