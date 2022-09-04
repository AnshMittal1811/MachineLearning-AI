CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch --nproc_per_node=2 \
train_distributed.py \
--batchsize 8 \
--savepath "../model/PGNet_DUT+HR" \
--datapath "../data/DUTS-TR+HR" \


