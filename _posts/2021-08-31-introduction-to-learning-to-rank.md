---
title: "How Search Result Rankings Are Calculated: Learning to Rank"
date: 2021-08-31
categories: [machine-learning]
tags: [search-algorithm, learning-to-rank, information-retrieval]
translation_key: introduction-to-learning-to-rank
---

There are countless documents on the web, and we can now search for almost any information that exists in the world. That makes a different question more important: "How do I find the information I want among all of that information?"

To think about it simply, I could ask a search engine to show me every document containing the keyword `Plato`. But would that really be a good search experience? If I had to read every document one by one to find information about Plato, it might be faster to email a philosophy professor instead.

What we need, then, is **a way to rank search results**. Among the many documents containing what I searched for, the system should show the best-written and most likely useful documents in order.

Seen this way, we have already identified the core of **Information Retrieval** fairly well.

1. Find documents containing the search terms
2. Define what "most useful" means
3. Calculate rankings according to that criterion

## Basic Principles of Search

Before going into details, let's first look at how search works. To simplify the explanation, I will refer to the various components of a search engine collectively as the "search bot."

### Extracting Index Terms from Documents

Before search can begin, there must first be data to show as results. Crawlers collect various documents from websites and store them in a database. At this point, the content of each document is also analyzed and stored.

One important piece of information in a document is words. For example, if we want to find documents containing the word `Plato`, it is much more efficient to store in advance which documents are connected to the word `Plato` than to scan every document in the database each time.

| Word | Documents |
| --- | --- |
| Plato | document1, document2, ... |
| Nietzsche | document2, document3, ... |

This structure, which connects words to documents, is called an **inverted index**. To extract words from documents for indexing, morphological analysis and stopword removal are also needed.

### User Queries and Intent

The user now enters what they want to know into the search box. Examples include `Plato biography`, `Korea Olympic schedule`, or `good shoes for jogging`. This is called a **query**.

To provide more accurate results, the search bot tries to understand the **user intent**. For example, if someone searches for `Korea Olympic schedule` while the Tokyo Olympics are taking place, it would be more appropriate to show Korea's event schedule for the Tokyo Olympics than the schedule for the PyeongChang Olympics held in Korea.

Also, as voice search has become more active, it has become important to handle not only simple keyword searches but also natural language queries such as `When was Plato born?`

### Database Search and Ranking

Once the user's query and intent are obtained, the search bot first uses the inverted index to retrieve candidate documents. For example, if the query is `good shoes for jogging`, it retrieves documents containing words such as `jogging`, `shoes`, and `good`.

It then calculates rankings by combining factors such as user intent, document credibility, and relevance between the query and document. The quality of this ranking calculation strongly affects the search experience.

## Hey Google, Learn to Rank

**Learning to Rank (LTR)** is also called **Machine-Learned Ranking (MLR)**. As discussed earlier, statistical information about keywords in a query is not enough to create good search results. Various features such as click counts, document credibility, freshness, and relevance to user intent must be extracted, and the optimal ranking must be learned from them.

An LTR model is generally built through the following process.

1. Create a judgment list
   - Match suitable documents to a given query.
2. Define features
   - Decide which features the model will learn from, such as click count, likes, document length, or title matching score.
3. Create training data
   - Set feature values for each document included in the judgment list.
4. Train and evaluate the model
   - Precision: the proportion of results returned by the model that are actually relevant.
   - Recall: the proportion of all relevant results that the model returned.
   - nDCG: a metric that evaluates search result quality while considering rank.
5. Apply it to the search engine

## nDCG: Normalized Discounted Cumulative Gain

Models learn in the direction of reducing error. There are several ways to evaluate the quality of a search ranking model, but here we will look at a representative metric, **nDCG (Normalized Discounted Cumulative Gain)**.

![nDCG formula](/assets/images/posts/2021-08-31-introduction-to-learning-to-rank/ndcg.png){: width="300" }

`DCG_p` calculates relevance for the top `p` search results while discounting the weight according to rank. Users usually look at higher-ranked search results more often, so the relevance of the first result is more important than the relevance of the hundredth result. DCG reflects this property.

However, since recommendation models or search models may return different result ranges, normalization is needed for comparison. Dividing `DCG_p` by `IDCG_p` gives the normalized value, nDCG. Here, `IDCG_p` is the DCG when the top `p` search results are ordered ideally.

Higher nDCG values indicate better search results.

## Learning to Rank Approaches

To return ordered search results, let's define a function `f`. `f(d, q)` takes document `d` and query `q` as input and returns the document's score or rank. The goal is to learn a function such that nDCG is maximized when all documents are sorted by `f(d, q)`.

LTR can be approached broadly in three ways: pointwise, pairwise, and listwise.

### Pointwise Learning to Rank

As the simplest example, consider the following formula.

```text
f(d, q) = 10 * titleScore(d, q) + 2 * descScore(d, q)
```

This example comes from [Search as Machine Learning](https://opensourceconnections.com/blog/2017/08/03/search-as-machine-learning-prob/). It calculates scores for all documents, then sorts them in descending score order.

The pointwise approach looks at each document individually and learns by reducing the difference between the calculated score and the target score. It is easy to understand and relatively simple to implement. However, if the error for the first-ranked document and the error for the hundredth-ranked document are treated in the same way, it becomes difficult to sufficiently reflect the greater importance of top-ranked results in real search.

### Pairwise Learning to Rank

The pairwise approach compares pairs of documents to adjust rankings. Given a pair of documents `(x_i, x_j)`, if `x_i` ranks higher than `x_j`, it can be assigned `1`; if lower, `-1`.

The fact that `x_i` ranks higher than `x_j` can be interpreted as meaning that we can classify which document is more relevant based on the difference between their features. Based on this idea, **RankSVM** finds a decision boundary that separates document pairs and learns ranking direction from it.

The pairwise approach has the advantage of learning relative order between documents. However, because it does not directly optimize the quality of the entire list, a gap can appear between the evaluation metric and the training objective.

### Listwise Learning to Rank

The listwise approach compares the ideal order of the entire document list with the order returned by the model. For example, the order from rank 1 to rank 100 can be considered one permutation among `100!` possible permutations. This method calculates and compares the probability that the search result permutation returned by the model is the actual target permutation.

When calculating permutations, it considers position-specific probabilities such as `the probability that document i is ranked first` and `the probability that document j is ranked second`. Therefore, it can give greater influence to higher rankings.

However, calculating ranking probabilities for all documents is computationally expensive. For this reason, simplified methods such as **Top-one probability** are sometimes used instead of calculating the full permutation.

## Summary

Search does not end with simply finding documents that contain keywords. It also requires calculating rankings for candidate documents so that users can find the information they actually want more quickly.

Learning to Rank is an approach that does not leave ranking calculations only to manually written rules, but instead lets a model learn from various features and evaluation data. Pointwise predicts the score of a single document, pairwise learns relative order between document pairs, and listwise directly handles the order of the entire search result list.

Ultimately, a good search system must solve both "finding documents" and "sorting documents well." Learning to Rank is one representative method for handling the second problem.

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

- Google Search Blog: <https://blog.google/products/search/>
- PageRank: <http://infolab.stanford.edu/~backrub/google.html>
- Crawling and indexing: <https://developers.google.com/search/docs/advanced/crawling/overview?hl=ko>
- Knowledge Graph: <https://blog.google/products/search/introducing-knowledge-graph-things-not/>
