---
title: "자연어 처리를 위한 텍스트 전처리 기법"
date: 2025-09-17
categories: [NLP, Machine Learning]
tags: [nlp, text processing, preprocessing]
---

자연어 처리에서 텍스트 전처리는 모델 성능에 큰 영향을 미치는 중요한 단계입니다.

## 기본 전처리 기법

### 1. 토큰화 (Tokenization)
- 문장을 단어 단위로 분리
- 공백, 구두점 기준으로 분할

### 2. 정규화 (Normalization)
- 대소문자 통일
- 특수문자 제거
- 숫자 처리

### 3. 불용어 제거 (Stop Words Removal)
- 의미가 없는 단어들 제거
- "그", "을", "를", "이", "가" 등

## 고급 전처리 기법

### 1. 형태소 분석
- 한국어의 경우 형태소 단위로 분리
- KoNLPy, Mecab 등 활용

### 2. 어간 추출 (Stemming)
- 단어의 어간만 추출
- "running" → "run"

### 3. 표제어 추출 (Lemmatization)
- 사전을 기반으로 기본형 추출
- "better" → "good"
