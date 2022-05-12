#!/bin/bash
echo -n "Train Data Collection, Validation Data Collection, Behavior Cloning or DAgger, or Evaluation? [data-train, data-val, data-expert, bc, dagger,eval] "
read MODE
echo -n "Enter Scenario No. [6,8,10]: "
read SCEN
export CUDA_VISIBLE_DEVICES=0
# Things to pay attention to
# for both BC and Dagger
# 1. Change --data to specify where the training data is stored

# For Dagger
# Make sure you start CARLA instances with correct port in a tmux pane
# ./scripts/launch_carla.sh [GPU_ID, 0] [NUM_WORKERS, 1] [PORT,2001] must be 2001 for now, hardcoded....
# Make sure mosquitto process are killed before running, although the script cleans by default after its own running

# 1. Change --finetune as the model that you want to continue training from BC
# 2. Change --benchmark_config to specify the dagger sampling configurations
# 3. I sample both data with collider and without collider, remove as you wish

# Must collect train and validation data before training

DATAFOLDER=~/AutoCast_{SCEN}
BATCHSIZE=32
WORKERS=32
CARLA_WORKERS=1
FRAMESTACK=1
TRAIN_SIZE=12
VAL_SIZE=12
DAGGER_SIZE=${CARLA_WORKERS}
BGTRAFFIC=60
if [[ $SCEN == 6 ]]
then
  BGTRAFFIC=30
fi
if [[ $SCEN == 8 ]]
then
  BGTRAFFIC=30
fi
if [[ $SCEN == 10 ]]
then
  BGTRAFFIC=30
fi

TrainValFolder=~/AutoCast_${SCEN}
DATAFOLDER=~/AutoCast_${SCEN}


if [[ $MODE == data-train ]]
then
AGENT=AutoCastSim/AVR/autocast_agents/simple_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${TrainValFolder}/Train/
CARLA_WORKERS=1
kill $(pgrep Carla)
kill $(pgrep ray)  
kill $(pgrep mosquitto)
./scripts/launch_carla.sh 0 ${CARLA_WORKERS} 2001 &
sleep 2

python3 AutoCastSim/parallel_scenario_runner.py  \
  --agent $AGENT \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic $BGTRAFFIC \
  --num-workers $CARLA_WORKERS \
  --file --sharing\
  --benchmark_config $CONFIG \
  --commlog  \
  --full \
  --emualte \
  --hud \
  --passive_collider \
  --outputdir $OUTPUTDIR \
  --resample-config 'random_uniform' \
  --num-config $TRAIN_SIZE
fi

if [[ $MODE == data-val ]]
then

AGENT=AutoCastSim/AVR/autocast_agents/simple_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${TrainValFolder}/Val/
CARLA_WORKERS=1
kill $(pgrep Carla)
kill $(pgrep ray)  
kill $(pgrep mosquitto)
./scripts/launch_carla.sh 0 ${CARLA_WORKERS} 2001 &
sleep 2

python3 AutoCastSim/parallel_scenario_runner.py  \
  --agent $AGENT \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic $BGTRAFFIC \
  --num-workers $CARLA_WORKERS \
  --file --sharing \
  --benchmark_config $CONFIG \
  --commlog  \
  --full \
  --emualte \
  --hud \
  --passive_collider \
  --outputdir $OUTPUTDIR \
  --resample-config 'random_uniform' \
  --num-config $VAL_SIZE
fi

if [[ $MODE == data-expert ]]
then
SEED=2
CUDA_VISIBLE_DEVICES=0
AGENT=AutoCastSim/AVR/autocast_agents/simple_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${TrainValFolder}/expert_seed${SEED}/
CARLA_WORKERS=1
kill $(pgrep Carla)
kill $(pgrep ray)  
kill $(pgrep mosquitto)
./scripts/launch_carla.sh ${CUDA_VISIBLE_DEVICES} ${CARLA_WORKERS} 2001 &
sleep 2

python3 AutoCastSim/parallel_scenario_runner.py  \
  --agent $AGENT \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic $BGTRAFFIC \
  --num-workers $CARLA_WORKERS \
  --file --sharing \
  --passive_collider \
  --benchmark_config $CONFIG \
  --commlog  \
  --emualte \
  --hud \
  --outputdir $OUTPUTDIR \
  --resample-config 'fixed' \
  --seed $SEED 
fi

if [[ $MODE == bc ]]
then
WORKERS=32
BATCHSIZE=32
#################### BC
#Test Input: Shared Lidar Voxel Output: Control
python3 -m training.train_v2v \
  --num-epochs 101 \
  --data $TrainValFolder/Train/ \
  --batch-size $BATCHSIZE \
  --num-dataloader-workers $WORKERS \
  --init-lr 0.0001 \
  --num-steps-per-log 100  \
  --frame-stack $FRAMESTACK \
  --max_num_neighbors 3\
  --device 'cuda' \
  --project 'cvpr-v2v'\
  --eval-data $TrainValFolder/Val/ 
fi

if [[ $MODE == dagger ]]
then
# Make sure you kill all carla processes
#################### DAgger
BATCHSIZE=32
RUN=${SCEN}-v2v-bc-w-ground-transform-run0
CHECKPOINT=wandb/${RUN}/files/model-100.th
BETA=0.8
CONFIG=benchmark/scene${SCEN}.json
CARLA_WORKERS=1
WORKERS=32
DAGGER_SIZE=1
DATAFOLDER=~/AutoCast_${SCEN}_Small/
kill $(pgrep CarlaUE4)
kill $(pgrep ray)  
kill $(pgrep mosquitto)

python3 -m training.train_dagger_v2v \
  --num-epochs 106 \
  --data $DATAFOLDER/Train/ \
  --daggerdata $DATAFOLDER/Dagger/ \
  --num-workers $CARLA_WORKERS \
  --batch-size $BATCHSIZE \
  --num-dataloader-workers $WORKERS \
  --init-lr 0.0001 \
  --num-steps-per-log 100  \
  --device 'cuda' \
  --finetune $CHECKPOINT \
  --beta $BETA --sampling-frequency 5 --checkpoint-frequency 5  \
  --benchmark_config $CONFIG \
  --bgtraffic $BGTRAFFIC \
  --max_num_neighbors 3 \
  --project 'cvpr-v2v' \
  --resample-config 'random_uniform' \
  --num-config $DAGGER_SIZE\
  --eval-data $TrainValFolder/Val/ 
fi


if [[ $MODE == eval ]]
then
#################### Evaluation
RUN=${SCEN}-v2v-dagger-run0
BGTRAFFIC=30
CHECKPOINTITER=105
CUDA_VISIBLE_DEVICES=0
SEED=0
AGENTCONFIG=wandb/${RUN}/files/config.yaml
AGENT=NeuralAgents/dagger_agent.py
CONFIG=benchmark/scene${SCEN}.json
CARLA_WORKERS=1
#kill $(pgrep Carla)
#kill $(pgrep ray)  
#kill $(pgrep mosquitto)
#./scripts/launch_carla.sh ${CUDA_VISIBLE_DEVICES} ${CARLA_WORKERS} 2001 &
#sleep 2

for NUMRUN in 0 1 2
do
for BGTRAFFIC in 0 15 30 45 #0 15 30 45
do
for SEED in 0 1 2
do
RUN=${SCEN}-v2v-dagger-run${NUMRUN}
AGENTCONFIG=wandb/${RUN}/files/config.yaml
OUTPUTDIR=${DATAFOLDER}/eval_${RUN}_bgtraffic${BGTRAFFIC}_seed${SEED}/
kill $(pgrep Carla)
#kill $(pgrep ray)  
#kill $(pgrep mosquitto)
python3 parallel_evaluation.py  \
  --agent $AGENT \
  --agentConfig $AGENTCONFIG \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic $BGTRAFFIC \
  --num-workers $CARLA_WORKERS \
  --file \
  --sharing \
  --emualte \
  --hud \
  --benchmark_config $CONFIG \
  --num_checkpoint $CHECKPOINTITER \
  --beta 0.0 \
  --passive_collider \
  --outputdir $OUTPUTDIR \
  --resample-config 'fixed' \
  --seed $SEED \
  --cuda_visible_devices $CUDA_VISIBLE_DEVICES 
done
done
done
fi
