#python train_lidar_and_meta.py --log_dir . --data ~/Documents/AutoCast_dataset --ego-only --batch-size 8 --num-dataloader-workers 8 --lr 0.001

#shared A*, shared lidar when training
#python train_lidar_and_meta.py --log_dir ./lidar_and_goal --data ~/Documents/AutoCast_shared_dataset --ego-only --batch-size 16 --num-dataloader-workers 16 --lr 0.05 --shared True 

#shared A*, non shared lidar when training
#python train_lidar_and_meta.py --log_dir ./liadr_and_goal_non_share --data ~/Documents/AutoCast_shared_dataset --ego-only --batch-size 16 --num-dataloader-workers 16 --lr 0.05 --shared False

#non-shared A*, shared lidar when training
#python train_lidar_and_meta.py --log_dir . --data ~/Documents/AutoCast_nonshared_dataset --ego-only --batch-size 8 --num-dataloader-workers 8 --lr 0.1 --shared True

#non-shared A*, non-shared lidar when training
#python train_lidar_and_meta.py --log_dir . --data ~/Documents/AutoCast_nonshared_dataset --ego-only --batch-size 8 --num-dataloader-workers 8 --lr 0.1 --shared False

#Lidar-only Model
#shared A*, shared lidar when training
python train_lidar.py --log_dir ./lidar_only_fixed --data ~/Documents/AutoCast_shared_dataset --ego-only --batch-size 64 --num-dataloader-workers 16 --lr 0.05 --shared True 
#python train_lidar.py --log_dir ./lidar_only_non_shared --data ~/Documents/AutoCast_shared_dataset --ego-only --batch-size 64 --num-dataloader-workers 16 --lr 0.05 --shared False 


