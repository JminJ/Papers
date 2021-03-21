# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding #
-----
## Abstract

 Bidirectional Encoder Representations from Transformers(BERT)는 당시의 language presentation model들과 다르게 unlabel된 text에서 모든 레이어의 왼쪽, 오른쪽 방향의 문맥을 통해 deep bidirectional representation을 학습하게 구성되어 있다. 

 그 결과로 BERT는 단 하나의 output layer를 사용해 fine-tunning을 하고 state-of-the-art 모델을 만들 수 있다. 

## Introduction

 pre-train이 Language model에 효과적인 성능 향상을 시켜준다는 것은 꾸준히 알려져 왔었다. 그렇다면 pre-train된 language representation을 적용시키는 방법에는 무엇이 있을까? 아래에 적어놓아보았다.

- feature-based model
- fine-tunning model

### feature-based model

대표적으로 ELMo가 있다. ELMo는 추가적인 feature로 사용하는 pre-train representation을 포함하는 task-specific architecture를 사용한다.

### fine-tunning model

대표적으로 GPT가 있다. GPT를 간략히 설명하자면 pre-train된 parameter들로 간단한 fine-tunning을 통해 downstream task들을 훈련시킨다.

 위 두가지 방법들은 pre-train동안 같은 object function을 공유한다(general language representation을 학습하기 위해 unidirectional language model을 사용한다)  

 저자들은 당시의 기술이 특히 fine-tunning에서 pre-train된 representation의 힘을 제한한다고 추측했다. 일반적인 language model의 주된 한계는 unidirectional하다는 것이다. 그리고 이는 pre-train될 때의 architecture를 제한한다는 문제를 가지고 있다. 이런 제한들은 sentence-level task들에는 차선책이 되나, 양 방향의 context를 통합하는 것이 중요한 question answering과 같은 token-level task에서는 매우 해롭게 작용할 수 있다. 

 따라서 이 논문에서 저자들은 fine-tuning을 기반으로하는 방법인 BERT를 제안한다. BERT는 MLM(masked language model) pre-training objective를 사용해 unidirectionality constraint를 완화한다. pre-train을 하는 동안 두 가지 방법을 사용해 학습을 한다.

- random하게 input의 token을 mask하고 mask된 원래의 vocabulary를 예측하는 것을 목표(objective)로 한다. 이는 왼쪽과 오른쪽의 문맥 정보를 합칠 수 있게 해준다.
- "next sentence prediction"이라 불리며 text-pair representation을 함께 학습하게 해준다.

## BERT

 BERT는 pre-training과 fine-tuning의 두 단계로 이루어져 있다. 먼저 pre-training 동안 model은 모두 제각각인 task들인 unlabeled data를 가지고 학습된다. fine-tuning동안에는 BERT는 이미 pre-train된 parameter들을 가지고 있을 것이다. 그리고 downstream task에 사용되는 labeled data를 사용해 fine-tune을 한다. 각각의 downstream task는 설령 같은 pre-train된 parameter를 사용한다고 해도 fine-tune된 model들마다 다르다. 

### Model Architecture

  BERT의 model architecture는 original Transformer에 기초한 multi-layer bidirectional Transformer encoder를 사용한다. 

 이 작업들에서 우리는 layer들의 수를 L으로, hidden size를 H로, self-attention head의 수를 A로 표현한다. 주로 아래 두 가지 모델 크기에 대한 결과를 보여줄 것이다.

- BERT_BASE : L=12, H=768, A=12, Total Parameters=110M
- BERT_LARGE : L=24, H=1024, A=16, Total Parameters=340M

BERT_BASE의 크기는 예전 GPT에서 본 기억이 날 수도 있다. BERT_BASE는 GPT와의 비교를 하기 위해 같은 크기로 만들어졌다. 그러나 GPT와 다른 점은 bidirectional self-attention을 사용했다는 점이다.

### Input/Output Representation

 BERT가 다양한 downstream task를 다룰 수 있게 만들기 위해 input representation은 하나의 token sequence에서 단일 문장과 한 쌍의 문장(<Question, Answer>)을 모호하지 않게 표현할 수 있다. 이 작업을 통해, 하나의 문장은 실제 언어적 문장이 아닌 무작위의 연속적인 text의 공간으로 표현할 수 있다. 

 30000개의 token vocabulary로 WordPiece embedding을 수행하였다. 모든 sequence의 첫 번째 token은 ([CLS])이다. 이 token의 마지막 hidden state는 classification task를 위해 집계 sequence representation으로 사용된다. 문장 쌍의 경우는 하나의 sequence로 합쳐진다. 두 문장을 구분하는 방법은 두 가지 방법이 있다.

- special token([SEP])을 사용해 나눠주는 것
- 문장 A인지, B인지 나타내는 학습된 embedding을 모든 token에 추가하는 것

<img src = '/images\2021_03_20_01.png' width='100%'>

Figure 1에서 볼 수 있듯, input embedding을 E로, special [CLS] token의 마지막 hidden vector를 C ∈ R_H라고 표현하고, i번째 input token의 마지막 hidden vector를 T_i ∈ R_H로 나타낸다.

  주어진 token에 대해, input representation은 그에 해당하는 token, segment, position embedding의 합으로 구조화된다. 이를 Figure 2에서 볼 수 있다.

### Pre-training BERT

 BERT를 훈련시키기 위해 기존의 left-to-right 또는 right-to-left language model들을 사용하지 않는다. 대신, 두 unsupervised task들을 사용해 BERT를 훈련시킨다. 이 방법들은 Figure 1의 왼쪽에 그려져 있다.

**Task #1: Masked LM**

 직관적으로 deep bidirectional model은 unidirectional model들과 두 방향의 model을 얕게 연결해놓은 model들보다 훨씬 성능이 좋다는 것을 알 수 있다. 안타깝게도 표준 conditional language model들은 왼쪽에서 오른쪽 또는 오른쪽에서 왼쪽으로만 학습이 가능하다. bidirectional 조건화를 통해 각 단어가 간접적으로 자신을 볼 수 있고, 모델이 multi-context에서 대상 단어를 사소하게 예측할 수 있기 때문이다.

 deep bidirectional representation을 학습하기 위해 input token에서 random하게 mask를 시킨다. 그 후, mask된 token을 예측한다.

 **Task #2: Next Sentence Prediction (NSP)**

 Question Answering(QA)과 Natural Language Inference(NLI)같이 중요한 downstream task들은 두 문장 사이의 relationship을 이해하는 것에 기반을 둔다. 그러나 language modeling은 이를 직접적으로 수집하지 못한다. 문장 사이의 relationship을 이해하는 model을 학습시키기 위해 monolingual corpus에서 사소하게 생성될 수 있는 binarized next sentence prediction task를 pre-train하였다. 

 문장 A와 B를 각 pre-training 예제에서 고르고 50%의 경우, A 뒤에 따르는 next sentence가 B로 사용된다(IsNext라고 labeling된다). 나머지 50%의 경우에는 corpus 내의 무작위 문장을 B로 사용한다(NotNext라고 labeling된다). Figure 1의 C가 NSP를 위해 사용된다(label). 이 방법은 매우 쉽게 느껴질 수 있으나 pre-train을 할 때 QA와 NLI task 모두 효과적인 모습을 볼 수 있었다.

<img src = '/images/2021_03_22_01.png'>

NSP task는 이전 연구들에서 사용된 representation learning objective에 매우 연관되어있다. 그러나 이전 연구들의 경우, 오직 sentence embedding만 downstream task로 전송된다. 여기서 BERT는 end-task model parameter들을 초기화하기 위해 모든 parameter들을 전송한다.

### Pre-training data

 Pre-training procedure은 기존의 language model pre-train 문헌을 따른다. pre-training corpus는 BooksCorpus(800M words)과 English Wikipedia(2500M words)를 사용하였다. Wikipedia의 경우 오직 text 구절만을 남겨두고 list, table, header들은 모두 삭제하였다. 긴 연속 sequence를 추출하기 위해 무작위로 섞인 sentence-level corpus보다 document-level corpus를 사용하는 것이 더욱 효과적이다. 

### Fine-tunning BERT

 text 쌍을 포함하는 애플리케이션의 경우 일반적인 패턴은 bidirectional cross attention을 적용하기 전에 text 쌍을 독립적으로 encodding하는 것이다. BERT는 대신 self-attention mechanism을 사용하여 이 두 단계를 통합한다. 연결된 text 쌍을 self-attention과 encodding하면 두 문장 사이의 bidirectional cross attention이 효과적으로 포함되기 때문이다.

 각 task에서 우리는 간단히 task specific input과 output을 BERT에 연결하고 모든 parameter들을 end-to-end fine-tune하면 된다. pre-training의 input sentence A와 B는 (1)paraprasing, (2)수반되는 hypothesis-premise 쌍, (3)question answering의 question-passage 쌍, (4) text를 생성하지 않는 text classification 또는 sequence tagging text-∅ 쌍과 유사하다. output에서, token represetation은 sequence tagging 또는 question answering같은 token level task에 입력이 된다. 그리고 [CLS] representation은 entailment 또는 sentiment analysis같은 classification output layer에 input이 된다. pre-training과 fine-tuning을 비교하면, fine-tuning은 상대적으로 저렴하다. 

## Experiment

### GLUE

<img src = '/images/2021_03_22_02.png'>

Table 1은 GLUE 벤치마크를 사용해 모델들을 평가한 표이다. BERT_BASE와 GPT의 성능차이를 보면 모든 task에서 우위를 점하고 있다는 것을 알 수 있다. 그리고 BASE model과 LARGE model의 성능 또한 매우 큰 차이를 보이는 것을 확인할 수 있다. 

 batch size는 32, fine-tune은 3epoch 동안 모든 GLUE task의 data에 대해 학습하였다. 각 task마다 최적의 learning rate(5e-5, 4e-5, 3e-5, and 2e-5)를 Dev set에 사용하였다. 추가적으로 BERT_LARGE의 경우 때때로 작은 dataset을 학습할 때 불안정한 모습을 보이기도 하였다. 

### SQuAD v1.1

 Wikipedia에서 주어지는 question, passage, answer를 사용해 passage 내의 answer text span을 예측하는 task이다. Figure 1에서 볼 수 있듯, question answering task에서, 우리는 input question과 passage를 single packed sequence로 표현한다(question은 A embedding을 사용하고, passage는 B embedding을 사용한다). 오직 start vector S ∈ R_H와 end vector E ∈ R_H 만을 fine-tuning 동안 소개한다. answer span의 start word i의 확률은 T_i와 S 사이에서 dot product 연산을 하고 뒤따르는 paragraph내의 모든 word들에 대한 softmax를 거쳐 연산한다: 

<img src = '/images/2021_03_22_03.png'>