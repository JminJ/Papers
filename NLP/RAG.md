# RAG
## Introduction
RAG(Retriebal-Augmented Generation)은 기존 QA task등의 모델들의 고질적인 문제였던 memory(pretrain을 거치며 학습된 지식 정보)를 수정, 확장이 불가능하다는 점과 hallucination(환각)을 생성한다는 점을 non-parametic memory(i.e., retrieval-based)를 통해 보완할 수 있다. 

[image1](../images/RAG_image1.png)

non-parametric memory는 wikipedia 데이터의 벡터 인덱스(vector index)로, pre-train된 retriever가 접근한다(*retriever가 wikipedia passage를 input으로 벡터 인덱스를 생성함*). non-parametric memory와 seq2seq model(generator)를 합쳐 end-to-end로 학습되는 probabilistic model로 만든다. 즉, retriever(DPR, Dense Passage Retriever)가 input과 연관된 latent document들을 추려주고 seq2seq model은 이를 input과 함께 사용하여 output을 생성한다.ß

Latent document들을 top-K approximation으로 marginalize 하는데 per-output basis, per-token basis 둘 중 한가지 방법으로 진행된다.

* per-output basis : 모든 토큰을 동일한 document를 가지고 추정(생성)한다.
* per-token basis : 다른 document들이 다른 token들을 추정하는데 사용된다.