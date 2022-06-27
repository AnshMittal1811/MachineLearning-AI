# Train teacher model
python teacher_train.py --config-name $1
# Distill experts from teacher and finetune
python experts_train.py --config-name $1

