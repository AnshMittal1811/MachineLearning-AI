#!/bin/bash
#RUN=6-v2v-dagger-run0
BGTRAFFIC=30
CHECKPOINTITER=105
CUDA_VISIBLE_DEVICES=0
SEED=0
AGENTCONFIG=ckpts/config.yaml
AGENT=NeuralAgents/dagger_agent.py
#AGENT=AutoCastSim/AVR/autocast_agents/simple_agent.py
CONFIG=benchmark/scene6.json
CARLA_WORKERS=2
NUMRUN=0
kill $(pgrep Carla)
kill $(pgrep mosquitto)
python3 parallel_evaluation.py  \
  --agent $AGENT \
  --agentConfig $AGENTCONFIG \
  --reloadWorld  \
  --port 2001 \
  --trafficManagerPort 3123 \
  --mqttport 4884 \
  --bgtraffic $BGTRAFFIC \
  --num-workers $CARLA_WORKERS \
  --hud \
  --emulate \
  --file \
  --sharing \
  --benchmark_config $CONFIG \
  --num_checkpoint $CHECKPOINTITER \
  --beta 0.0 \
  --passive_collider \
  --resample-config 'fixed' \
  --seed $SEED \
  --cuda_visible_devices $CUDA_VISIBLE_DEVICES 
