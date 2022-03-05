# joint-intent-classification-and-slot-filling-based-on-BERT
BERT for joint intent classification and slot filling


About BERT, please read this paper: BERT:   
https://www.aclweb.org/anthology/N19-1423 (pre-training of deep bidirectional transformer for language understanding);  
https://github.com/google-research/bert (official Github website about BERT)

About this project, please read this paper:  
https://arxiv.org/pdf/1902.10909.pdf (BERT for joint intent classification and slot filling)

For training:   
python train.py --train=data/atis/train --val=data/atis/valid --save=saved_models/atis_max50_drop_ep30 --epoch=30 --batch_size=128

For evaluating:  
python evaluate.py --model=saved_models/atis_max50_drop_ep30 --data=data/atis/test --batch=128 --pre_intents=pre_intents --pre_slots=pre_slots
