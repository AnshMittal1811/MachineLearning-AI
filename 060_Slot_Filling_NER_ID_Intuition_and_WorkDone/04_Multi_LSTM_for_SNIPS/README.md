# multiLSTM for Joint NLU

this implements a recurrent model for joint intent detection and slot-filling for the NLU task.

## requirements

```
gensim==3.4.0
h5py==2.8.0
Keras==2.2.0
keras-contrib==2.0.8
keras-utilities==0.5.0
numpy
tensorflow==1.9.0
```

for installation of `keras-contrib` see [their repo](https://github.com/keras-team/keras-contrib)

## updates and todo

this is a complete rewrite using ATIS dataset, see the `old_version` branch for the original version

currently working on using the 2017 benchmark from snips:  
https://github.com/snipsco/nlu-benchmark

please be aware that i am not super-active on github and i provide my code in an 'as-is' state. as this is not written to be used for commercial purposes, i am not concerned about bug-fixing pull requests as much as comments regarding theoretical issues.

## files

- `00_data_processing-test.ipynb` for processing ATIS data
- `01_keras_modeling-test.ipynb` for training model
- `02_keras_demo.ipynb` for decoding new sentences on trained model
- `snips_00_preprocessing.ipynb` for processing SNIPS data

## background

A conversational spoken dialog system (SDS) traditionally consists of a pipeline of multiple elements operating in sequence; a common pipeline consists of *automatic speech recognition* (ASR) or *text-to-speech* (TTS), *natural language understanding* (NLU), a *dialog management system* (DMS) or *state tracking* module,*natural language generation* (NLG) and *text-to-speech* (TTS). *Natural Language Understanding*, also referred to as *Spoken Language Understanding* (SLU) when used as a component in an SDS, can be considered the act of converting a user's natural language request to a compact format that that the dialog manager can use to decide the proper response.

For *task-oriented* conversational agents ("chatbots"), the slot-filling paradigm is frequently used as seen in Lui & Lane 2016 and Goo *et al.* 2018, among others. This approach breaks down the NLU task into two primary sub-tasks, *slot-filling* (SF) and *intent detection\/determination* (ID). 

In this approach, a user's preferences are assigned to a number of **slots**, which the dialog manager attempts to fill with the **value** of the user's preference (in a programming analogy, a slot is a variable and the value is a variable's assigned value). These slots can be completely open, restricted to a set of types (such as days of the week, restaurant types, flight classes such as *first class* or *economy*), or boolean in value.

Along with slots, the NLU must also classify **intent**. Intent captures the goal or purpose of a user's utterance. It can be roughly thought of as analogous to a programming *function* that takes a number of slots as arguments. Together, the intent and slots provide a summary of the user's utterance that can be used to determine how the chatbot responds.

The slot-filling and intent detection tasks can be handled separately, for example using a sequence model such as a Markov Model or conditional random field (CRF) for slot filling word-by-word, and a SVM or logistic regression model over the entire utterance for intent classification. However, with the growth of deep neural networks, "Joint NLU" has become a popular approach, in which a single network detects both intent and slot values. This allows for a simpler system (as it uses a single machine-learned model for both tasks), and allows for varying degrees of synergy in training the model. In some cases this may just mean using a single input for both tasks, but some networks may use the predictions of one task to assist with the other.

## purpose

This model demonstrates a Frankenstein example of a joint NLU model, borrowing concepts from a number of current papers on joint SLU/NLU and the related named entity recognition (NER) task. It makes use of convolutional sub-word embeddings + pretrained word embeddings, a biLSTM-CRF sequence model for sequence tagging plus an attention-based weighted vector for intent classification. For details, see the model training notebook.

## model

this section will briefly explain the overall model architecture and the rationale behind some of the model choices.

the model accepts two inputs representing one sentence: a 1D array of integer-indexed word-level tokenized input, e.g. `[hello, world, this, is, a, test]` and 2D array of per-word character-level input: `[[h,e,l,l,o], [w,or,l,d],...,[t,e,s,t]]`. Ma & Hovy 2016 note that research has found that using multiple input methods, e.g. distributed word embeddings & engineered features, outperform single input method models. for their NER model, they use word-level embeddings and subword character-level CNN embeddings, which we adopt here. intuitively, we may hope that the word-level embeddings, trained via context, can detect a mixture of syntactic and semantic features, e.g. *scholar* has a high degree of 'nouniness' and appears related to other academic words; while the character-level embeddings, by using convolutional networks of various widths, may focus on detecting n-gram patterns such as 'schola' that may relate the word to words with similar patterns such as 'scholastic', 'scholarship' etc. also, because the set of characters is relatively constrained compared to the set of possible words, this may help the network recognize unseen words by their subword features alone. we encourage this by using a more aggressive dropout on the word-level embeddings. we also pass the subword embeddings through a highway layer, as proposed by Yoon Kim in his character-based CNN-LSTM language model as an alternative to a simple feed-forward network (or none at all). this layer adds a gating function that controls information flow and is *roughly* analogous to things like the resnet architecture, in that it is primarily used for training very deep networks. it's included here because of the essential ml research principle, the Rule of Cool.

this results in each word being represented by two 1D vectors, (for example) a word vector of size 200 and a subword vector of size 100, which are stacked head-to-toe to create a single vector of size 300, so the sentence is represented as a sequence of 300-feature vectors. a recurrent network 'reads' this sequence and outputs its output state, or what we can intuitively consider its 'memory' of the sentence up to that point, at each step of the sequence. we use a bidirectional LSTM, in which we have two recurrent networks reading the sequence, one backwards and one forwards. we stack the memory vectors head-to-toe much like we did the word and subword vectors, such that we have one single LSTM output vector for each word, which now represents a sort of 'local memory' at that point *extending both forward and backward*, again *loosely* analogous to the human process of reading, in which [saccadic eye movements](https://en.wikipedia.org/wiki/Eye_movement_in_reading#Saccades) track context on *both sides* of the 'fixation point'.

Liu & Lane extend the recurrent layer into a deep recurrent layer using 'aligned sequence-to-sequence'. as explained above, the recurrent layer takes in a sequence of input features and is configured here to output a sequence of output features corresponding to the input at each timestep. we can imagine taking these outputs and using them as inputs to another recurrent layer to further abstract the representations. Liu & Lane initialize the second recurrent layer's states with the final states of the previous layer, providing another source of information to the second layer: it starts out with a 'memory' of what the last layer had read. this is 'aligned' because the *inputs* are the first layer's outputs, forcing the second layer's outputs to align with the original sequence length, unlike standard sequence-to-sequence where the input and output sequences do not need to align in length. so we can do this too using `return_state` to get the first layer's final states, and `initial_state` to initialize the second layer's states. like Liu & Lane, we swap the forward and backward states.

while recurrent networks sort of 'implicitly' model sequential information based on their memory cells, modern named entity recognition models often use explicit sequential information for modeling output sequences; in cases where we have a large distributed feature vector as input, models like the maximum-entropy Markov model or [conditional random field](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf) are favored. while other joint SLU researchers Goo *et al.* achieve good results with using the raw LSTM states to obtain marginal probabilities at each timestep, and other papers have experimented with other methods such as using viterbi or beam decoding (rather than greedy); because we can treat slot detection as a type of entity recognition, we add a CRF based on previous NER research, in particular Huang, Xu and Yu 2015 and Ma & Hovy 2016.

Liu & Lane; Goo *et al.* also discuss how previous joint SLU models don't condition one task on the other; they simply use the LSTM output as input to two branches of the network, one that does sequence modeling for slot-filling and one that does intent recognition over the entire sentence. each takes a different approach in regard to which task is conditioned on the other; here we use Liu & Lane's approach of conditioning the intent prediction on the slot predictions. we do this by using the CRF predictions as input to the attention layer used to predict the intent.

the attention layer weights each input according to how well it thinks that input affects the task (in this case, intent detection). intuitively, this would mean 'stopwords' such as 'a', 'the', etc. would be considered less important, and context words would be weighted more heavily. we use the attention with context layer from `keras-utilities` based on another of Hovy's papers, *Hierarchical Attention Networks for Document Classification*. this outputs a single sentence vector that consists of the attention-weighted sums of the input vector sequence: in this case, the concatenated decoder LSTM output and CRF output. this sentence vector is then passed through a feedforward network for intent detection.

## dataset

the ATIS dataset can be obtained here:  
https://github.com/yvchen/JointSLU

## results

our intent accuracy is **96.3%**, which beats out Goo *et al.* 2018's reported score of 94.1%. 

our slot F1 score of 93.71 using the phrase-based `conlleval` approach, substantially lower than their 95.2. of course we can tune a lot of things here or try the greedy LSTM.

note: the `conlleval` python script is not included, see **licensing** for link to source file.

```
# TRAIN RESULTS

INTENT F1 :   0.9958062755602081  (weighted)
INTENT ACC:   0.9966502903081733
SLOT   F1 :   0.9943244981289089  (weighted, labels only)

# TEST RESULTS

INTENT F1 :   0.9573853410886499  (weighted)
INTENT ACC:   0.9630459126539753
SLOT   F1 :   0.9508826933073504  (weighted, labels only)
CONLLEVAL :   93.71
```

## examples

we can use the decoding script to load the saved model and weights, and decode new sentences:

```
query: looking for direct flights from Chicago to LAX
slots:
{'connect': 'direct', 'fromloc.city_name': 'chicago', 'toloc.city_name': 'lax'}
intent: atis_flight

query: give me flights and fares from New York to Dallas
slots:
{'fromloc.city_name': 'new york', 'toloc.city_name': 'dallas'}
intent: atis_flight#atis_airfare

query: i want a first class flight to los angeles
slots:
{'class_type': 'first class', 'toloc.city_name': 'los angeles'}
intent: atis_flight
```

## references

Goo *et al* (2018): *Slot-Gated Modeling for Joint Slot Filling and Intent Prediction*  
NAACL-HCT 2018, available: http://aclweb.org/anthology/N18-2118

Hakkani-Tur *et al* (2016): *Multi-Domain Joint Semantic Frame Parsing using Bi-directional RNN-LSTM*  
available: https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_MultiJoint.pdf

Kim *et al* (2015): *Character-Aware Neural Language Models*  
available: https://arxiv.org/pdf/1508.06615.pdf

Liu & Lane (2016): *Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling*  
INTERSPEECH 2016, available: https://pdfs.semanticscholar.org/84a9/bc5294dded8d597c9d1c958fe21e4614ff8f.pdf

Ma & Hovy (2016): *End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF*  
available: https://arxiv.org/pdf/1603.01354.pdf

Park & Song (2017): *음절 기반의 CNN 를 이용한 개체명 인식 Named Entity Recognition using CNN for Korean syllabic character*  
available: https://www.dbpia.co.kr/Journal/ArticleDetail/NODE07017625 (Korean)

Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). *Highway networks*.  
available: https://arxiv.org/pdf/1505.00387.pdf

## licensing

i don't want to commit to a license but in the spirit of the ATIS corpus and the above papers, i provide this code for research purposes. an attribution would be nice (Derek Hommel) if you do something cool, and if you want to collaborate on a project or paper, feel free to contact me at dsh9470@snu.ac.kr.

models may use layers from [keras-contrib](https://github.com/keras-team/keras-contrib) released under the MIT License

models may use layers from [keras utilities](https://github.com/cbaziotis/keras-utilities) by cbaziotis released under the MIT License

models may use layers written *with reference to* [keras2-highway-network.py](https://gist.github.com/iskandr/a874e4cf358697037d14a17020304535) by iskandr

phrase-based slot F1 score may be determined using a modified [conlleval python port](https://github.com/sighsmile/conlleval) by sighsmile (*not included due to lack of license*) or using Goo *et al.*'s `conlleval` [based script](https://github.com/MiuLab/SlotGated-SLU) in `utils.py` 

