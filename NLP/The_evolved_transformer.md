# The Evolved Transformer
------
## Introduction
Reinforcement learning(강화학습)을 활용한 기법이 사람들이 만든 모델들보다 성능이 높은 일이 자주 있었다. sequence model을 searching하는 것에 대한 많은 노력이 있었지만 Vision model들에 이점들이 집중됬다. 

그러나 최근 연구에서 sequence problem을 해결하기 위한 RNN의 대안이 나왔다. Convolution Seq2Seq 같은 convolution model의 성공과 Transformer 같은 full attention network의 성공 때문에 feed-forward network들이 이제는 seq2seq task를 해결하기 위한 필수적인 요소가 되었다.

이러한 feed-forward model보다 더 좋은 성과를 얻기 위해 tournament selection과 Transformer로 warm start를 하는 neural architecture search method를 만들었다. 앞서 말한 성과를 달성하기 위해 최근의 feed-forward seq2seq model들의 성과와 *Progressive Dynamic Hurdles*(PDH) 기법을 사용하는 search space를 구조화했다. 

우리의 연구는 *Evolved Transformer*라는 새로운 architecture를 만들어 냈다. model size가 큰 경우에는 WMT'14 En-De에서 BLEU score 29.8을 달성해 state-of-the-art를 달성했다. 또한 더 작은 size일 떄도 성능이 뛰어난다, 기존 big Transformer보다 37.6% 정도 적은 parameter를 가지고 Transformer보다 BLEU score가 0.7 더 높았다.
## Methods
