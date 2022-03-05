# Intent Model
Bi-directional LSTM with attention for intent detection and entity slot filling

## Requirements
tensorflow 1.4 <br>
python 3.5


## How does it work?
The model needs 3 input files:
* label - contains the intent label for the utterance
* seq.in - contains the utterance itself
* seq.out - contains the ground truth mapping for the utterance; you will see the nouns replaced by the entity labels

The model is a bi-directional LSTM with attention that generates a hidden forward state and a hidden backwards state for 
an utterance. It also learns a mapping for an entity label using a context vector that is just the weighted sum of the 
hidden states multiplied by an attention weighting vector. An intent context vector is also computed using only the 
final hidden state from the LSTM. Both entity filling and intent detection use a softmax activation function. The entity 
filling model can also utilize a tanh gate enabling the user to optimize for just intent rather than solving the joint 
optimization problem. The default setting solves the joint problem with attention on both the intent and the entities. 
The --model_type option lets you run it with attention only on intent.

The code takes in the input data and create dictionaries for both intent and entity labels based on seq.out file
It writes them to a vocab folder.


## Usage

* run with 32 nodes on the atis dataset and no patience for early stop: <br>
&emsp;python3 train.py --num_units=32 --dataset=atis --patience=0

* run with 64 nodes on atis data with early stopping disabled and using only intent attention: <br>
&emsp;python3 train.py --no_early_stop --dataset=atis --model_type=intent_only


## Best Results



Test data set: <br>
Slot F1: 96.5 <br>
intent accuracy: 96.2 <br>
semantic error: 86.2 <br>

For comparison see the Carnegie Mellon paper: https://arxiv.org/pdf/1609.01454.pdf
