# TextAugmentation-GPT2
![GPT2 model size representation](https://github.com/prakhar21/TextAugmentation-GPT2/blob/master/gpt2-sizes.png)
Fine-tuned pre-trained GPT2 for topic specific text generation. Such system can be used for Text Augmentation.

## Getting Started
1. git clone https://github.com/prakhar21/TextAugmentation-GPT2.git
2. Move your data to __data/ dir__.

_* Please refer to data/SMSSpamCollection to get the idea of file format._

## Tuning for own Corpus
1. Assuming are done with Point 2 under __Getting Started__
```
2. Run python3 train.py --data_file <filename> --epoch <number_of_epochs> --warmup <warmup_steps> --model_name <model_name> --max_len <max_seq_length> --learning_rate <learning_rate> --batch <batch_size>
```
## Generating Text
```
1. python3 generate.py --model_name <model_name> --sentences <number_of_sentences> --label <class_of_training_data>
```

_* It is recommended that you tune the parameters for your task. Not doing so may result in choosing default parameters and eventually giving sub-optimal performace._

## Quick Testing
I had fine-tuned the model on __SPAM/HAM dataset__. You can download it from [here](https://drive.google.com/open?id=1lDMFdcSsmWuzHIW8ceEgDnuJHzxX8Hiw) and follow the steps mentioned under __Generation Text__ section.

_Sample Results_
```
SPAM: You have 2 new messages. Please call 08719121161 now. £3.50. Limited time offer. Call 090516284580.<|endoftext|>
SPAM: Want to buy a car or just a drink? This week only 800p/text betta...<|endoftext|>
SPAM: FREE Call Todays top players, the No1 players and their opponents and get their opinions on www.todaysplay.co.uk Todays Top Club players are in the draw for a chance to be awarded the £1000 prize. TodaysClub.com<|endoftext|>
SPAM: you have been awarded a £2000 cash prize. call 090663644177 or call 090530663647<|endoftext|>

HAM: Do you remember me?<|endoftext|>
HAM: I don't think so. You got anything else?<|endoftext|>
HAM: Ugh I don't want to go to school.. Cuz I can't go to exam..<|endoftext|>
HAM: K.,k:)where is my laptop?<|endoftext|>
```

## Important Points to Note
* _Top-k and Top-p Sampling_ (Variant of __Nucleus Sampling__) has been used while decoding the sequence word-by-word. You can read more about it [here](https://arxiv.org/pdf/1904.09751.pdf)


__Note:__ First time you run, it will take considerable amount of time because of the following reasons - 
1. Downloads pre-trained gpt2-medium model  _(Depends on your Network Speed)_
2. Fine-tunes the gpt2 with your dataset _(Depends on size of the data, Epochs, Hyperparameters, etc)_

All the experiments were done on [IntelDevCloud Machines](https://software.intel.com/en-us/devcloud)
