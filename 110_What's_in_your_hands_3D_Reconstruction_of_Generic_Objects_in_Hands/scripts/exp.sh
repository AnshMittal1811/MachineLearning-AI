python -m models.ihoi --config experiments/obman.yaml  --slurm EXP dev


eval:
python -m models.ihoi   --eval --ckpt /checkpoint/yufeiy2/hoi_output/release_model/obman/checkpoints/last.ckpt  --slurm 

python -m models.ihoi   --eval --config experiments/ho3d.yaml --ckpt /checkpoint/yufeiy2/hoi_output/release_model/ho3d/checkpoints/last.ckpt  --slurm

python -m models.ihoi   --eval --config experiments/mow.yaml --ckpt /checkpoint/yufeiy2/hoi_output/release_model/mow/checkpoints/last.ckpt   --slurm


train: 
python -m models.ihoi --config experiments/obman.yaml  --slurm 


python -m models.ihoi --config experiments/mow.yaml  --ckpt /checkpoint/yufeiy2/hoi_output/release_model/obman/checkpoints/last.ckpt --slurm

python -m models.ihoi --config experiments/ho3d.yaml  --ckpt /checkpoint/yufeiy2/hoi_output/release_model/obman/checkpoints/last.ckpt --slurm


-
python -m models.ihoi --config experiments/obman.yaml  --slurm 

python ihoi.py --config experiments/pifu.yaml MODEL.BATCH_SIZE 32 \


 python -m demo.demo_hoi -e /checkpoint/yufeiy2/hoi_output/release_model/obman