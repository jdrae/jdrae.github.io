---
title: "검색 결과의 순위를 계산하는 법: Learning to Rank"
date: 2021-08-31
permalink: /ko/2021/08/31/introduction-to-learning-to-rank/
categories: [machine-learning]
tags: [search-algorithm, learning-to-rank, information-retrieval]
translation_key: introduction-to-learning-to-rank
---

웹에는 수많은 문서가 있고, 우리는 이제 세상에 존재하는 거의 모든 정보를 검색할 수 있게 되었습니다. 그래서 더 중요해진 질문은 "그 많은 정보 중에서 내가 원하는 정보를 어떻게 찾을 것인가"입니다.

단순하게 생각하면, 내가 검색한 `플라톤`이라는 키워드를 포함하는 문서를 전부 보여달라고 할 수 있습니다. 하지만 그것이 정말 좋은 검색일까요? 모든 문서를 하나하나 읽으며 플라톤에 대한 정보를 찾을 바에는, 차라리 철학과 교수님께 이메일을 보내는 편이 더 빠를지도 모릅니다.

그래서 필요한 것이 **검색 결과의 순위를 매기는 방법**입니다. 내가 검색한 내용을 포함하는 수많은 문서 중에서 가장 잘 쓰였고, 가장 도움이 될 가능성이 높은 문서를 순서대로 보여주는 것입니다.

이렇게 보면 우리는 이미 **정보 검색(Information Retrieval)** 의 핵심을 꽤 잘 짚은 셈입니다.

1. 검색어를 포함하는 문서를 찾기
2. "가장 도움이 되는 것"이 무엇인지 정의하기
3. 그 기준에 맞게 순위를 계산하기

## 검색의 기본 원리

세부적인 내용을 다루기 전에, 검색이 어떤 과정으로 이루어지는지 먼저 살펴보겠습니다. 여기서는 설명을 단순화하기 위해 검색엔진의 여러 구성 요소를 통틀어 "검색봇"이라고 부르겠습니다.

### 문서의 색인어 추출

검색을 시작하기 전에는 먼저 결과로 보여줄 데이터가 있어야 합니다. 크롤러는 웹사이트의 다양한 문서를 수집하고 데이터베이스에 저장합니다. 이때 문서의 내용도 함께 분석해 저장합니다.

문서에서 중요한 정보 중 하나는 단어입니다. 예를 들어 `플라톤`이라는 단어가 들어간 문서를 찾으려면, 데이터베이스에 있는 모든 문서를 매번 하나씩 훑는 것보다 `플라톤`이라는 단어와 연결된 문서를 미리 저장해두는 편이 훨씬 효율적입니다.

| 단어 | 문서 |
| --- | --- |
| 플라톤 | 문서1, 문서2, ... |
| 니체 | 문서2, 문서3, ... |

이렇게 단어에서 문서로 연결되는 구조를 **역색인(inverted index)** 이라고 합니다. 문서에서 색인으로 사용할 단어를 추출하기 위해서는 형태소 분석과 불용어 제거도 필요합니다.

### 사용자의 질의와 의도

이제 사용자는 검색창에 알고 싶은 내용을 입력합니다. `플라톤 생애`, `한국 올림픽 일정`, `조깅할 때 좋은 신발` 같은 검색어가 여기에 해당합니다. 이를 **질의(query)** 라고 합니다.

검색봇은 더 정확한 결과를 보여주기 위해 **사용자의 의도(user intent)** 를 파악하려고 합니다. 예를 들어 도쿄 올림픽이 진행되는 시점에 `한국 올림픽 일정`이라고 검색했다면, `한국에서 열렸던 평창 올림픽 일정`보다 `도쿄 올림픽에서의 한국 경기 일정`을 보여주는 편이 더 적절할 것입니다.

또한 음성 검색이 활발해지면서 단순한 키워드 검색뿐 아니라 `플라톤이 언제 태어났어?` 같은 자연어 질의를 처리하는 것도 중요해졌습니다.

### DB 검색과 순위 계산

사용자의 질의와 의도를 얻으면, 검색봇은 먼저 역색인을 이용해 후보 문서를 추립니다. 예를 들어 `조깅할 때 좋은 신발`이라는 질의가 들어오면, `조깅`, `신발`, `좋은` 같은 단어가 포함된 문서를 전체 문서에서 가져옵니다.

그다음 사용자의 의도, 문서의 신뢰도, 질의와 문서의 관련성 등을 종합해 순위를 계산합니다. 이 순위 계산의 품질이 검색 경험을 크게 좌우합니다.

## Hey Google, Learn to Rank

**Learning to Rank(LTR)** 는 **Machine-Learned Ranking(MLR)** 이라고도 합니다. 앞서 살펴봤듯이 좋은 검색 결과를 만들기 위해서는 질의에 쓰인 키워드의 통계적 정보만으로는 부족합니다. 클릭 횟수, 문서 신뢰도, 최신성, 사용자 의도와의 관련성처럼 다양한 특징(feature)을 추출하고, 이를 바탕으로 최적의 순위를 학습해야 합니다.

LTR 모델은 대체로 다음과 같은 과정으로 만들어집니다.

1. 평가 리스트(judgment list) 만들기
   - 주어진 질의에 적절한 문서를 매칭합니다.
2. 특징(feature) 정의하기
   - 클릭 횟수, 좋아요 수, 문서 길이, 제목 매칭 점수 등 모델이 학습할 특징을 정합니다.
3. 훈련 데이터 만들기
   - 평가 리스트에 포함된 문서별 특징 값을 설정합니다.
4. 모델 훈련 및 평가하기
   - Precision: 모델이 반환한 검색 결과 중 실제로 관련 있는 결과의 비율입니다.
   - Recall: 전체 관련 결과 중 모델이 반환한 관련 결과의 비율입니다.
   - nDCG: 순위까지 고려해 검색 결과 품질을 평가하는 지표입니다.
5. 검색엔진에 적용하기

## nDCG: Normalized Discounted Cumulative Gain

모델은 오차를 줄이는 방향으로 학습합니다. 검색 랭킹 모델의 품질을 평가하는 방법은 여러 가지가 있지만, 대표적으로 **nDCG(Normalized Discounted Cumulative Gain)** 를 살펴보겠습니다.

![nDCG 공식](/assets/images/posts/2021-08-31-introduction-to-learning-to-rank/ndcg.png){: width="300" }

`DCG_p`는 상위 `p`개의 검색 결과를 순위에 따라 가중치를 줄여가며 관련도를 계산합니다. 보통 사용자는 상위 검색 결과를 더 많이 보기 때문에, 1위 문서의 관련성이 100위 문서의 관련성보다 더 중요합니다. DCG는 이런 특성을 반영합니다.

다만 추천 모델이나 검색 모델마다 결과 범위가 다를 수 있으므로, 서로 비교하기 위해 정규화가 필요합니다. `DCG_p`를 `IDCG_p`로 나누면 정규화된 값인 nDCG를 얻을 수 있습니다. 여기서 `IDCG_p`는 상위 `p`개 검색 결과가 이상적인 순서로 정렬되었을 때의 DCG입니다.

nDCG는 값이 클수록 더 좋은 검색 결과를 의미합니다.

## Learning to Rank 접근 방법

순서가 있는 검색 결과를 반환하기 위해 함수 `f`를 정의해보겠습니다. `f(d, q)`는 문서 `d`와 질의 `q`를 입력받아 문서의 점수 또는 순위를 반환합니다. 목표는 모든 문서를 `f(d, q)`에 따라 정렬했을 때 nDCG가 최대화되도록 함수를 학습하는 것입니다.

LTR은 크게 pointwise, pairwise, listwise 방식으로 접근할 수 있습니다.

### Pointwise Learning to Rank

가장 단순한 예로 다음과 같은 공식을 생각할 수 있습니다.

```text
f(d, q) = 10 * titleScore(d, q) + 2 * descScore(d, q)
```

이 예시는 [Search as Machine Learning](https://opensourceconnections.com/blog/2017/08/03/search-as-machine-learning-prob/)에서 소개된 방식입니다. 모든 문서의 점수를 계산한 뒤, 점수가 높은 순서대로 정렬합니다.

Pointwise 방식은 각 문서를 하나씩 보고, 계산된 점수와 목표 점수의 차이를 줄이는 방식으로 학습합니다. 이해하기 쉽고 구현도 비교적 단순합니다. 다만 1위 문서의 오차와 100위 문서의 오차를 같은 방식으로 다룬다면, 실제 검색에서 더 중요한 상위권 결과의 가중치를 충분히 반영하기 어렵습니다.

### Pairwise Learning to Rank

Pairwise 방식은 두 문서의 쌍(pair)을 비교하며 순위를 조정합니다. 문서 `x_i`와 `x_j`의 쌍 `(x_i, x_j)`가 있을 때, `x_i`가 `x_j`보다 순위가 높으면 `1`, 낮으면 `-1`처럼 값을 매길 수 있습니다.

`x_i`가 `x_j`보다 순위가 높다는 것은, 두 문서의 특징 차이를 통해 어느 문서가 더 관련성이 높은지 분류할 수 있다는 뜻으로 볼 수 있습니다. 이런 아이디어를 바탕으로 **RankSVM**은 문서 쌍을 구분하는 결정 경계를 찾고, 이를 통해 순위 방향을 학습합니다.

Pairwise 방식은 문서 간 상대적인 순서를 학습할 수 있다는 장점이 있습니다. 다만 전체 리스트의 품질을 직접 최적화하는 것은 아니기 때문에, 평가 지표와 학습 목표 사이에 차이가 생길 수 있습니다.

### Listwise Learning to Rank

Listwise 방식은 전체 문서 목록의 이상적인 순서와 모델이 반환한 순서를 비교합니다. 예를 들어 1위부터 100위까지의 순서는 `100!`개 중 하나의 순열로 볼 수 있습니다. 모델이 반환한 검색 결과 순열이 실제 목표 순열일 확률을 계산해 비교하는 방식입니다.

순열을 계산할 때는 `1위가 문서 i일 확률`, `2위가 문서 j일 확률`처럼 위치별 확률을 고려합니다. 따라서 상위권 순위에 더 큰 영향을 줄 수 있습니다.

다만 전체 문서의 순위 확률을 계산하는 것은 계산량이 많습니다. 그래서 전체 순열 대신 **Top-one probability**처럼 단순화된 방식으로 계산하기도 합니다.

## 정리

검색은 단순히 키워드가 포함된 문서를 찾는 문제에서 끝나지 않습니다. 사용자가 실제로 원하는 정보를 더 빠르게 찾도록, 후보 문서의 순위를 계산하는 과정이 필요합니다.

Learning to Rank는 이 순위 계산을 사람이 직접 만든 규칙에만 맡기지 않고, 다양한 feature와 평가 데이터를 바탕으로 모델이 학습하게 하는 접근입니다. Pointwise는 문서 하나의 점수를 예측하고, pairwise는 문서 쌍의 상대적 순서를 학습하며, listwise는 전체 검색 결과 리스트의 순서를 직접 다룹니다.

결국 좋은 검색 시스템은 "문서를 찾는 것"과 "문서를 잘 정렬하는 것"을 함께 해결해야 합니다. 그리고 Learning to Rank는 그 두 번째 문제를 다루는 대표적인 방법입니다.

---

## References

- <https://www.google.com/search/howsearchworks/>
- <https://elasticsearch-learning-to-rank.readthedocs.io/en/latest/core-concepts.html>
- <https://opensourceconnections.com/blog/2017/02/24/what-is-learning-to-rank/>
- <https://opensourceconnections.com/blog/2017/08/03/search-as-machine-learning-prob/>
- <https://www.youtube.com/watch?v=eMuepJpjUjI&ab_channel=Lucidworks>
- <https://lucidworks.com/post/abcs-learning-to-rank/>
- <https://ride-or-die.info/normalized-discounted-cumulative-gain/>

## More to Read

- 구글 서치 블로그: <https://blog.google/products/search/>
- 페이지랭크: <http://infolab.stanford.edu/~backrub/google.html>
- 크롤링과 색인: <https://developers.google.com/search/docs/advanced/crawling/overview?hl=ko>
- 지식 그래프: <https://blog.google/products/search/introducing-knowledge-graph-things-not/>
