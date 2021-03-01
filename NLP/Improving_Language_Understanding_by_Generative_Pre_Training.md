# Improving Language Understanding by Generative Pre-Training #
------
## Abstract

 Natural language understanding은 넓은 범위의 다양한 task들로 이루어져 있다(textual entailment, question answering, semantic similarity assessment and document classification). unlabeled text corpora는 매우 풍부한 반면, 특정한 task를 훈련하기 위해 필요한 labeled data는 매우 부족하다. 이것이 알맞는 성능을 가지기 위해 훈련되는 모델들의 난관이다. 따라서 우리는 각 특정한 task에 대한 discriminative fine-tuning이 따라오고 다양한 unlabeled text의 corpus로 훈련시킨 language model, generative pre-training을 통해 큰 성능의 향상이 가능하다는 것을 설명할 것이다.

이전 접근법들과는 다르게 모델의 architecture는 크게 바꾸지 않고 fine-tuning을 하는 동안 효과적인 transfer(전이)을 할 수 있게 task-aware input transformation을 사용한다. 우리는 우리 접근법의 다양한 natural language understanding benchmark들에 대한 효과성을 설명할 것이다. 우리 general task-agnostic model은 각 task을 위해 만들어진 architexture를 사용하며 개별적으로 훈련된 model보다 월등히 좋은 모습을 보여준다(12개의 task에서 9개의 state-of-the-art). 

## Introduction

raw text에서 효과적이게 배우는 능력은 Natural language processing에서 supervised learning의 의존성을 완화시키는데 매우 중요하다. 대다수의 deep learning 방법들은 상당한 양의 labeled data를 요구한다(주석처리 된 자원드르이 부족으로 인해 고통받는 많은 도메인에 대한 적용성을 제한한다). 이러한 상황 속, unlabeled data에서 언어적 정보를 활용할 수 있는 model들은 시간이 많이 걸리고 비용이 많이 드는 많은 주석(annotation)들을 모으는데 의미있는 차선책을 제공한다. 더 나아가서 supervision(supervised?)이 가능함에도 unsupervised fashion에서 좋은 representation들을 배우는 것은 상당한 성능 향상을 제공한다. 가장 설득력 있는 증거는 지금까지  pre-train된 word embedding들을 광범위하게 사용하는 것은 NLP task의 범위에서 성능향상을 보여 주었다는 것이다. 

그러나 unlabeled text에서 word-level보다 더 큰 정보를 활용하는 것은 두 개의 주요 이유로 어렵다. 첫 번째는 전이에 사용되는 text representation을 학습하는데 가장 효과적인 optimazation 목표의 type이 불 명확하다는 것이다. 최근의 연구는 language modeling, machine translation, discourse coherence와 같은 다양한 목표를 조사했으며, 각 방법은 다른 작업에서 다른 방법을 능가한다. 두 번째는, 배운 representation을 target task로 전이하는 가장 효과적인 방법에 연관점(consensus)이 없다는 것이다. 기존의 방법들은 복잡한 학습 체계를 사용하여 model architecture를 task별로 변경하고 learning scheme들과 보조 learning 목표를 추가하는 것의 조합을 포함한다. 이 불확실성들은 language processing에서 semi-supervised learning을 개발하는것에 어려움을 만들어 왔다.

이 논문에서 우리는 language understanding task에서 unsupervised pre-training과 supervised fine-tuning의 조합을 사용하는 semi-supervised 접근법을 탐구한다. 우리의 목표는 다양한 범위의 task에서 작은 적응과 함께 전이되는 보편적인 표현을 배우는 것이다. 수동으로 주석이 달린 training 예제를(target task) 사용해 unlabeled text와 여러 dataset들의 large corpus에 액세스하는 것으로 가정한다. 우리의 구조는 target task가 unlabeled corpus와 같은 도메인에 있어야 한다는 것을 요구하지 않는다. 우리는 2단계의 training 절차를 채택하고 있다. 첫 번째로 우린 neural network model의 initial parameter들을 배우기 위해 unlabeled data에 language modeling 목표를 사용한다. 그 다음, 우리는 그 parameter들을 supervised 목적에 해당하는 target task에 적용한다.

## Related work

**semi-supervised learning for NLP**

우리의 작업은 대체로 natural language를 위한 semi-supervised learning 범주에 속한다. 이 패러다임은 sequence labeling이나 text classification과 같은 task들에 적용하는 것에 큰 관심을 끌었다. 가장 처음에 사용된 방법은 supervised model에서 사용되는 feature의 word-level 혹은 phrase-level를 계산하기 위해 unlabeld data를 사용하는 것이였다. 이후 몇 년간, 연구자들은 다양한 task의 성능 향상을 위해 unlabeled data로 훈련된 word embedding 사용의 이점에 대해 설명했다. 이 접근법들은 주로 word-level 정보를 전이(transfer)하는데 반해 우리는 더 높은 level의 의미를 수집하는데 의미를 둔다.

최근의 접근법들은 word-level보다 더 높은 의미를 unlabeled data에서 배우거나 활용할 수 있게 연구되는 중이다. unlabeled corpus를 통해 훈련될 수 있는 phrase-level 혹은 sentence-lavel embedding은 다양한 target task에서 더 좋은 vector 표현을 encode하기 위해 사용되고 있다.

**Unsupervised pre-training**

Unsupervised pre-training은 semi-supervised learning에서 목표가 supervided learning 목표를 바꾸는 것 대신 좋은 시작(initialization) point를 찾는 특별한 케이스다. 이전 연구들은 image classification이나 regression task들에서 사용하는 것을 연구했지만 이후의 연구들은 regularization scheme와 같이 deep neual network에서 더 좋은 일반화(generalization)를 할 수 있을 pre-training act들을 설명한다. 최근의 연구에서는 image classification, speech recognition, entity disambiguation, machine translation와 같이 다양한 task에서 deep neural network의 학습을 돕기 위한 방법을 사용했다.

우리의 연구와 가장 가까운 연구는 language modeling objective에서 사용되는 neural network를 pre-training 하고 그것을 supervision(supervided)와 함께 target task에서 fine-tuning하는 것을 포함하는 것이다. Dai et al.과 Howard 그리고 Ruder는 text classification의 성능을 높이기 위해 위의 방법을 따랐다. 그러나 언어적 정보를 수집하는 것을 돕기 위해 pre-training 단계를 거친 반면에 LSTM model을 사용해 짧은 범위에서의 예측 능력을 제한시켜 버렸다. 이에 반해 우리의 transformer network 선택은 이 논문의 실험에서 설명하듯 긴 범위의 언어적 구조를 수집할 수 있게 해준다. 더 나아가 우리 모델의 넓은 범위의 task(natural language inference, paraphrase detection, story completion)에 대한 효과성 또한 설명할 것이다. 다른 접근법들은 pre-train 된 language 또는 machine translation model에서의 hidden state를 target task에서 supervised 학습을 하는 동안 보조 feature로 사용한다. 이는 각각 분리된 target task 마다 엄청난 양의 새로운 parameter들을 포함하게 된다. 하지만 우리는 전이(transfer)를 하는 동안 아주 작은 model architecture의 변화만을 요구한다.

**Auxiliary training objectives**

보조 unsupervised 학습 목표를 추가하는 것은 semi-supervised learning의 대안의 형태이다. 예전 연구에서는 entity recognition이라 불리는 POS tagging chunking같은 NLP task에서 사용하였다. (생략) . 아주 최근에 Rei는 language modeling 목표를 그들의 target task 목표에 추가하였고, sequence labeling task들에서 성능 향상을 설명하였다. 우리의 실혐들 또한 보조 목표들을 사용했다. 우리가 보여줄 것 처럼, unsupervised pre-training은 이미 target task들과 연관이 있는 언어적 측면들을 배울 수 있다.  

## Framework

우리의 절차는 두 가지 단계를 밟는다. 첫 번째는 많은 양의 text corpus에 대한 고용량 language model을 배우는 것이다. 이는 labeled data와 함께 각각의 task를 model에 적응시키는 fine-tuning이 따라온다. 

### Unsupervised pre-training

unsupervised corpus tokens = U = {u1, . . ., un}이 주어지면, 우리는 아래에 표현된 likelihood를 최대화 하기 위해 standard language modeling 목표를 사용한다.

<img src = '/images/2021_03_01_01.png'>

k는 context window의 크기이고, 조건부 확률 P는 Θ parameter들과 함께 neural network에서 사용되게 모델링 된다. 이 parameter들은 SGD(stochastic gradient descent)를 사용해 훈현이 된다.

우리의 실험에서, transformer의 변형인 multi-layer Transformer decoder를 language model로 사용했다. 이 모델은 input context token들에 대해 self-attention연산을 적용한 후 position-wise feedforward layer를 적용하여 target token에 대한 output 분포를 생성한다 :

<img src = '/images/2021_03_01_02.png'>

U = (u_-k, . . . u_-1)은 token들의 context vector이고, n은 layer들의 수 이다. W_e는 token embedding matrix(행렬)이고, W_p는 position embedding matrix이다.

### Supervised fine-tuning

Eq.(1)의 objective(목적)과 같이 model을 훈련시킨 후, 우리는 supervised target task에 대해 parameter들을 적응시켜야 한다. 우리는 labeled dataset을 C라고 가정한다. C는 각 요소들이 y와 함께 x^1, . . . ,x^m으로 구성된 input token들의 sequence로 구성된다.  input값들은 우리의 pre-train된 model을 통과함으로써 마지막 transformer block의 activation h^m_l을 얻는다(h^m_l은 parameter W_y와 함께 y를 예측하기 위해 추가된 linear output layer에 입력으로 사용된다) :

<img src = '/images/2021_03_01_03.png'>

이것은 우리에게 최대화 해야하는 following objective를 준다.

<img src = '/images/2021_03_01_04.png'>

우리는 추가적으로 language modeling을 보조 목표(auxiliary objective)로 추가하는 것이 아래 두 가지 이유로 fine-tuning의 학습을 도운다는 것을 발견했다.

- (a) supervised model의 일반화(generalization)을 향상시킨다.
- (b) 수렴을 가속화한다.

이는 보조 목표를 사용해 성능을 향상시켰던 이전 연구들과 일맥상통하다. 구체적으로 우리는 following objective를 λ weight와 함께 최적화 시켰다 :

<img src = '/images/2021_03_01_05.png'>

전체적으로, 오직 fine-tuning동안 우리가 요구하는 extra(추가) parameter들은 W_y와 구분 token들의 embedding들 뿐이다.(section "Task-specific input transformations"에서 설명한다)

<img src = '/images/2021_03_01_06.png' width = '95%'>

### Task-specific input transformations

text classicication과 같은 task들에서, 우리는 model을 위에서 설명한 것처럼 바로 fine-tune을 할 수 있다. question answering 또는 textual entailment 같은 다른 task들은 문장 순서 쌍, 또는 문서, 질문, 답변의 세 쌍으로 이루어진 구조화 된 input을 가지고 있다. 우리의 pre-train model은 text의 연속 sequence로 훈련되기 때문에, 우리는 이러한 task들에 적용하기 위해 조금의 수정이 필요하다. 이전의 연구는 전이된 representation의 위에  task specific architecture들을 학습하는 것을 제안해왔다. 이러한 접근 방식은 상당한 양의 task-specific 맞춤화를 재도입하고 이러한 추가 architectural 구성 요소에 대해 전이 학습을 사용하지 않는다. 대신, 우리는 구조화된 input들을 pre-train model이 처리할 수 있는 순서 sequence로 변환하는 traversal-stype 접근법을 사용했다. 이 input transformation은 여러 task에 걸쳐 architecture들의 광범위한 변화를 피할 수 있게 해준다. 우리는 이 input transformation에 대한 간략한 설명을 아래에 제공할 것이고 시각화를 Figure 1에 제공했다. 모든 transformation들은 무작위로 초기화 되는 시작과 끝 token(\<s>, \<e>)의 추가를 포함한다. 

**Textual entailment(두 개의 문장에서 첫 문장이 두 번째 문장을 수반하는가 혹은 위반하는가?)**

entailment task들에 대해 우리는 전제(premise) p와 가설(hypothesis) h token sequence들을 구분기호(delimiter) token($)을 사이에 끼고 연결시킨다(concatenate).

**Similarity**

similarity task들에 대해서는 비교될 두 문장간의 내재 순서가 존재하지 않는다. 이 사실을 반영하기 위해 우리는 input sequence를 가능한 문장 순서들을 모두 포함하게 수정했다(문장 사이에 구분기호가 들어간다) 그리고 두 sequence representation h^m_l(linear output layer에 입력으로 가기 전 element-wise로 더해지는 것)을 생산하기 위해 각각 독립적으로 처리한다. 

**Question Answering and Commonsence Reasoning**

이 task들에 대해서는 우리는 context document z, question q, 가능한 답변들의 set {a_k}가 주어진다. document와 question, 각 가능한 답변들을 구분기호 token을 사이에 넣어 [z;q;$;a_k]의 형태로 연결한다. 각 sequence들은 가능한 답변들에 해대 output 분포를 만들기 위해 softmax layer로 정규화되고 우리의 model을 통해 독립적으로 처리된다.