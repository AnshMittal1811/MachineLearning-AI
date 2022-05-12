#!/bin/bash
echo -n "Train Data Collection, Validation Data Collection, Behavior Cloning or DAgger, or Evaluation? [data-train, data-val, data-expert, bc, dagger,eval] "
read MODE
echo -n "Enter Scenario No. [6,8,10]: "
read SCEN
echo -n "Enter model type. [COOPERNAUT, V2V]:"
read MODEL
echo -n "Enter fusion type. [late, early, no]:"
read FUSION
export CUDA_VISIBLE_DEVICES=0

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

FUSIONFLAG="";
NPOINT="";
NBLOCK="";
CPT="";
TRANSFORMERDIM="";

TrainValFolder=./data/AutoCast_${SCEN}
DATAFOLDER=$TrainValFolder

if [[ $MODE == data-train ]]
then
AGENT=AutoCastSim/AVR/autocast_agents/simple_agent.py
CONFIG=benchmark/scene${SCEN}.json
OUTPUTDIR=${TrainValFolder}/Train/
kill -9 $(pgrep Carla)
kill -9 $(pgrep ray)
kill -9 $(pgrep mosquitto)
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
kill -9 $(pgrep Carla)
kill -9 $(pgrep ray)  
kill -9 $(pgrep mosquitto)
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
kill -9 $(pgrep Carla)
kill -9 $(pgrep ray)  
kill -9 $(pgrep mosquitto)
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

if [[ $MODEL == COOPERNAUT ]]
then
BCSCRIPT=training.train_point_transformer;
DAGGERSCRIPT=training.train_dagger_point_transformer;
NPOINT="--npoints 2048";
NBLOCK="--nblocks 2";
TRANSFORMERDIM="--transformer_dim 32";
fi

if [[ $MODEL == V2V ]]
then
BCSCRIPT=training.train_v2v;
DAGGERSCRIPT=training.train_dagger_v2v
fi

if [[ $FUSION == early ]]
then
FUSIONFLAG="--earlyfusion";
fi

if [[ $FUSION == no ]]
then
MAXNEIGHBOR=0;
fi

if [[ $FUSION == late ]]
then
MAXNEIGHBOR=3;
    if [[ $MODEL == COOPERNAUT ]]
    then
    CPT="--cpt";
    fi
fi
    
if [[ $MODE == bc ]]
then

python3 -m $BCSCRIPT \
  --num-epochs 101 \
  --data $TrainValFolder/Train/ \
  --batch-size $BATCHSIZE \
  --num-dataloader-workers $WORKERS \
  --init-lr 0.0001 \
  --num-steps-per-log 100  \
  --frame-stack $FRAMESTACK \
  --device 'cuda' \
  --project 'bc' \
  --eval-data $TrainValFolder/Val/ \
  $MAXNEIGHBOR \
  $TRANSFORMERDIM \
  $CPT \
  $NPOINT \
  $NBLOCK \
  $FUSIONFLAG
fi

if [[ $MODE == dagger ]]
then
RUN=${SCEN}-efpt-bc-run0
CHECKPOINT=wandb/${RUN}/files/model-100.th
DAGGER_SIZE=1
kill $(pgrep CarlaUE4)
kill $(pgrep ray)  
kill $(pgrep mosquitto)

python3 -m $DAGGERSCRIPT \
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
  --project 'dagger' \
  --resample-config 'random_uniform' \
  --num-config $DAGGER_SIZE \
  --eval-data $TrainValFolder/Val/ \
  $MAXNEIGHBOR \
  $TRANSFORMERDIM \
  $CPT \
  $NPOINT \
  $NBLOCK \
  $FUSIONFLAG
fi

if [[ $MODE == eval ]]
then
#################### Evaluation
RUN=${SCEN}-efpt-dagger-run0
BGTRAFFIC=30
CHECKPOINTITER=105
CUDA_VISIBLE_DEVICES=0
SEED=0
AGENTCONFIG=wandb/${RUN}/files/config.yaml
AGENT=NeuralAgents/dagger_agent.py
CONFIG=benchmark/scene${SCEN}.json

for NUMRUN in 0 1 2
do
RUN=${SCEN}-efpt-dagger-run${NUMRUN}
for BGTRAFFIC in 0 15 45 #0 15 30 45
do
for SEED in 0 1 2
do
OUTPUTDIR=${DATAFOLDER}/eval_${RUN}_bgtraffic${BGTRAFFIC}_seed${SEED}/
kill $(pgrep Carla)
#kill $(pgrep ray)  
#kill $(pgrep mosquitto)
python3 AutoCastSim/parallel_evaluation.py  \
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
  --commlog \
  --emulate \
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
