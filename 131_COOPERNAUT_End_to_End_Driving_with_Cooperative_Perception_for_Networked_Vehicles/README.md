# Coopernaut

![teaser](assets/introduction.svg)
> [**COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles**](https://ut-austin-rpl.github.io/Coopernaut/)    
> Jiaxun Cui*, Hang Qiu*, Dian Chen, Peter Stone, Yuke Zhu    
> _CVPR 2022_

[Paper](https://arxiv.org/abs/2205.02222) | [Website](https://ut-austin-rpl.github.io/Coopernaut/) | [Bibtex](#citation) | [Dataset](https://utexas.box.com/v/coopernaut-dataset)

## Introduction
Optical sensors and learning algorithms for autonomous vehicles have dramatically advanced in the past few years. Nonetheless, the reliability of today's autonomous vehicles is hindered by the limited line-of-sight sensing capability and the brittleness of data-driven methods in handling extreme situations. With recent developments of telecommunication technologies, cooperative perception with vehicle-to-vehicle communications has become a promising paradigm to enhance autonomous driving in dangerous or emergency situations.

We introduce COOPERNAUT, an end-to-end learning model that uses cross-vehicle perception for vision-based cooperative driving. Our model encodes LiDAR information into compact point-based representations that can be transmitted as messages between vehicles via realistic wireless channels. To evaluate our model, we develop [AutoCastSim](https://github.com/hangqiu/AutoCastSim), a network-augmented driving simulation framework with example accident-prone scenarios. Our experiments on AutoCastSim suggest that our cooperative perception driving models lead to a 40% improvement in average success rate over egocentric driving models in these challenging driving situations and a 5 times smaller bandwidth requirement than prior work V2VNet. 

## Installation

First, clone the Coopernaut repo with AutoCastSim submodule.

```bash
git clone --recursive git@github.com:UT-Austin-RPL/Coopernaut.git
```
We provide a quickstart installation script in `scripts/install.sh`. 
In the root folder of this repo, run 

```bash
source scripts/install.sh #Requires CUDA11.0
```
or
```bash
source scripts/install-cu113.sh #Requires CUDA11.3
```
If you encounter errors or would like to follow individual steps, please refer to [INSTALL.md](docs/INSTALL.md) for more details.

## Quick Start

We provide a quick example to create evaluation trajectories of a trained Coopernaut model under the scenario 6:Overtaking, with two parallel threads(Make sure your GPU has a memory larger than 6GB, otherwise change the `CARLA_WORKERS` in the `scripts/quick_run.sh` to `1`). You can check the saved trajectories under `Coopernaut/result` directory. 
```bash
conda activate autocast
./scripts/quick_run.sh
```

## Usage

Our scripts are separated based on models. For example, all data collection, DAgger, training and evaluation of Coopernaut codes are contained in the script `run_cooperative_point_transformer.sh`. 

```bash
bash run_cooperative_point_transformer.sh
```

To checkout a specific method, please replace the script with the method's corresponding pipeline script

|    Method    |    Pipeline Script  |
| :----------- | :----------------- |
| No V2V Sharing | `bash run_point_no_fusion.sh` |
| Early Fusion | `bash run_earlyfusion_point_transformer.sh` |
| Voxel GNN | `bash run_v2v.sh` |
| Coopernaut | `bash run_cooperative_point_transformer.sh` |

The below instructions apply to all scripts, and all the modification should be **in the scripts**.

First, **in the scripts**, specify the root path you wish to store behavior cloning data from: `TrainValFolder`, and the path to store DAgger data from: `DATAFOLDER`. By default they are

```vim
TrainValFolder=./data/AutoCast_${SCEN}
DATAFOLDER=./data/AutoCast_${SCEN}_Small
```

for each scene.


### Data Collection

[Download Dataset](https://utexas.box.com/v/coopernaut-dataset)

Before training, you may download the behavior cloning data [here](https://utexas.box.com/v/coopernaut-dataset), or collect your own data by running the scripts provided. In the first prompt, select `data-train`, `data-val` or `data-expert` and select the scene number:

| Scene number |        Scene        |
| :----------- | :-----------------: |
| 6            |      Overtake       |
| 8            |      Left turn      |
| 10           | Red light violation |

### Behavior cloning training

Run this after you have both training and validation data collected in the previous step. In the first prompt, select `bc`. 

### DAgger training

Specify the behavior cloning model checkpoint path trained in the previous stage  `CHECKPOINT`, and then when running script, select `dagger` in the prompt.

### Evaluation

[Download Pretrained Models](https://utexas.box.com/v/coopernaut-dataset) 

Specify `RUN=` to be the path of your DAgger model path. Then, select `eval` as the prompt.
Then get the metrics by running
```bash
python scripts/result_analysis.py --root ${PATH_TO_YOUR_OUTPUT_DIR}
python scripts/compare_trajectory.py --eval ${PATH_TO_YOUR_OUTPUT_DIR} --expert ${PATH_TO_EXPERT}
```

## Citation
```bibtex
@inproceedings{cui2022coopernaut,
    title = {Coopernaut: End-to-End Driving with Cooperative Perception for Networked Vehicles},
    author = {Jiaxun Cui and Hang Qiu and Dian Chen and Peter Stone and Yuke Zhu},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022}
}
```
