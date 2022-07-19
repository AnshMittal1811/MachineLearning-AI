#!/bin/sh

models=(beit_b16 effenetb4)

datasets=(activity aircraft)


for model in ${models[@]};
do
  echo $model
  for datast in ${datasets[@]};
  do
    echo $datast
    sh run_srun.sh 2 configs/100p/config_${datast}.yaml configs/models_cfg/${model}.yaml ${model}_100p_${datast}
    sleep 10
  done
done
