# ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS
-----
## Introduction

<img src = '/images/2021_07_08_01.png'>

 Electra는 같은 크기의 model, data, compute일 때 Bert나 XLNet같은 MLM-Model들과 비교해 큰 성능차이을 보여준다(Figure 1). 게다가 Electra는 모델 크기 면에서도 좋은 장점을 띄게 되는데,  Electra-small의 경우, 단일 GPU로 4일만에 학습이 완료되었고, RoBERTa와 XLNet과 비교 가능한 성능의 Electra-Large의 경우 그 model들보다 4배 작은 parameter들로 학습을 할 수 있다. 

## Method

<img src = '/images/2021_07_08_01.png'>
