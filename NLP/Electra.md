# ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS
-----
## Introduction

<img src = '/images/2021_07_08_01.png' width = '70%'>

 Electra는 같은 크기의 model, data, compute일 때 Bert나 XLNet같은 MLM-Model들과 비교해 큰 성능차이을 보여준다(Figure 1). 게다가 Electra는 모델 크기 면에서도 좋은 장점을 띄게 되는데,  Electra-small의 경우, 단일 GPU로 4일만에 학습이 완료되었고, RoBERTa와 XLNet과 비교 가능한 성능의 Electra-Large의 경우 그 model들보다 4배 작은 parameter들로 학습을 할 수 있다. 

## Method

<img src = '/images/2021_07_08_01.png' width = '70%'>

두 가지의 neural network를 훈련시키게 된다. 하나는 Generator G, 나머지 하나는 Discriminator D이다. 각각 Transformer의 encoder로 구성되어 있다(input token **x**  = [x_1,  ... , x_n]을 contextualized vector representations의 sequence h(**x**) = [h1, ... , h_n]으로 mapping한다). 주어진 position t는 오로지 x_t = [MASK]인 위치만을 가르키고, generator의 output들은 softmax layer를 사용해 particular token x_t를 만들기 위한 확률값이다.

<img src = '/images/2021_07_18_01.png' width = '70%'>

e는 token embedding을 의미한다. position t가 주어지면 discriminator은 token x_t(generator distribution에서 나온 data)가 진짜인지 아닌지를 판별한다(sigmoid output layer를 통해).

<img src = '/images/2021_07_18_02.png' width = '70%'>

genarator은 MLM을 수행하기 위해 학습되어진다. **x** = [x_1, ... ,x_n]이 주어지게 되면, MLM은 우선 무작위의 set들을 선택한다(1~n까지의 integer). 이후 선택된 position들을 **m** = [m_1, ... , m_k]라 하고, 이 position들을 [MASK] token으로 변환한다 : 우리는 이를 **x**^**masked** = **REPLACE**(**x**, **m**, ****[MASK])라 부른다. 이때 generator은 mask 처리된 token들의 원본 identity를 예측하는 식으로 학습하게 된다. discriminator은 genarator으로 변환된 token들로부터 나온 data의 token들을 구별하게 학습된다. 더 세밀하게 설명하면, 우리는 변형된 example인 **x**^**corrupted**를 masking된 token을 generator의 sample로 만들게 되고, discriminator는 원본 input인 **x**와 **x^corrupted**가 일치하는지 예측하는 것을 배우게 된다. model input들은 아래 설명을 따라 구성되게 된다.

<img src = '/images/2021_07_18_03.png' width = '70%'>

loss function은 위처럼 아래에 설명을 첨가하였다.

<img src = '/images/2021_07_18_04.png' width = '70%'>

GAN의 training objective와 유사하나 몇가지 다른 점들은 존재한다. 첫째, generator가 올바른 token을 생성했다면, 그 token은 "fake" 대신 "real"로 인식된다; 이 공식은 downstream task에서 적절한 성능향상을 가져왔다. 더욱 중요한 것은, generator은 discriminator를 속이기 위해 적대적(adversarially)으로 훈련되는 것이 아니라 maximum likelihood로 학습된다. Adversarially 학습 generator은 매우 어려운데, 이유는 generator로부터 sampling된 결과를 토대로 backpropagate하는 것은 불가능하기 때문이다. 이 문제점을 피하기 위해 generator를 학습할 때 reinforcement learning을 사용한다. 하지만 이는 maximum-likelihood training보다 좋지 못한다. 마지막으로, input에 noise vector를 넣어 generator에 적용시키지 않는다.

합쳐진 loss를 최소화 할 수 있다.

<img src = '/images/2021_07_18_05.png' width = '70%'>

raw text의 큰 corpus X에 대해서 구해진다. 단일 sample로 loss의 기대치를 근사화할 수 있다. 또한 우리는 discriminator의 loss를 generator를 통해 back-propagete하지 않는다(실제로, sampling 단계 때문에 할 수 없다). pre-training이 끝난 뒤, generator는 버리고, downstream task들에서 discriminator를 fine-tune한다.