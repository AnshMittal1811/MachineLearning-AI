#!/bin/bash
echo -n "Train Data Collection, Validation Data Collection, Behavior Cloning or DAgger, or Evaluation? [data-train, data-val, bc,dagger,eval] "
read MODE
echo -n "Enter Scenario No. [6,8,10]: "
read SCEN

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

#DATAFOLDER=/hdd1/AutoCast/Scen${SCEN}_bg120_passivecollider/
#DATAFOLDER=CoDrive_Dataset/Scen${SCEN}_bg120_passivecollider/
DATAFOLDER=~/AutoCast_{SCEN}
BATCHSIZE=32
WORKERS=8
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

TrainValFolder=/scratch/cluster/jxcui/Documents/AutoCast_${SCEN}
DATAFOLDER=/scratch/cluster/jxcui/Documents/AutoCast_${SCEN}


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

AGENT=AutoCastSim/AVR/autocast_agents/simple_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${TrainValFolder}/expert/
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
  --passive_collider \
  --benchmark_config $CONFIG \
  --commlog  \
  --emualte \
  --hud \
  --outputdir $OUTPUTDIR \
  --resample-config 'fixed'
fi

if [[ $MODE == bc ]]
then
BATCHSIZE=32
#################### BC
#Test Input: Shared Lidar Voxel Output: Control
python3 -m training.train_v2v \
  --num-epochs 51 \
  --data $TrainValFolder/Train/ \
  --batch-size $BATCHSIZE \
  --num-dataloader-workers $WORKERS \
  --init-lr 0.0001 \
  --num-steps-per-log 100  \
  --frame-stack $FRAMESTACK \
  --max_num_neighbors 0\
  --device 'cuda' \
  --project 'cvpr-nonshare-voxel'\
  --eval-data $TrainValFolder/Val/ 
fi

if [[ $MODE == dagger ]]
then
# Make sure you kill all carla processes
#################### DAgger
BATCHSIZE=32
RUN=${SCEN}-nonshare-voxel-bc
CHECKPOINT=wandb/${RUN}/files/model-50.th
BETA=0.8
CONFIG=benchmark/scene${SCEN}.json
CARLA_WORKERS=1
WORKERS=4
DAGGER_SIZE=1
DATAFOLDER=/scratch/cluster/jxcui/Documents/AutoCast_${SCEN}_Small/
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
  --max_num_neighbors 0 \
  --project 'cvpr-nonshare-voxel' \
  --resample-config 'random_uniform' \
  --num-config $DAGGER_SIZE\
  --eval-data $TrainValFolder/Val/ 
fi


if [[ $MODE == eval ]]
then
#################### Evaluation
RUN=${SCEN}-nonshare-voxel-dagger
CHECKPOINTITER=105

AGENTCONFIG=wandb/${RUN}/files/config.yaml
AGENT=NeuralAgents/dagger_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${DATAFOLDER}/eval_${RUN}_bgtraffic${BGTRAFFIC}/
CARLA_WORKERS=1
kill $(pgrep Carla)
kill $(pgrep ray)  
kill $(pgrep mosquitto)
./scripts/launch_carla.sh 0 ${CARLA_WORKERS} 2001 &
sleep 2

python3 AutoCastSim/parallel_scenario_runner.py  \
  --agent $AGENT \
  --agentConfig $AGENTCONFIG \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic $BGTRAFFIC \
  --num-workers $CARLA_WORKERS \
  --file \
  --commlog \
  --emualte \
  --hud \
  --benchmark_config $CONFIG \
  --num_checkpoint $CHECKPOINTITER \
  --beta 0.0 \
  --passive_collider \
  --outputdir $OUTPUTDIR\
  --resample-config 'fixed'

fi

