# Rendering multi-view RGB, polarization and normal images using Mitsuba2

Python code for rendering multi-view images in Mitsuba2 that comprise of RGB, polarization and normal channels. This code also saves the ground truth poses in different formats used in conventional NeRF/multi-view inverse rendering works. 

Useful for generating data to run the following neural radiance fields-based works: 

1. [NeRF](https://github.com/yenchenlin/nerf-pytorch)
2. [VolSDF](https://github.com/ventusff/neurecon) (tested with neurecon repo)
3. [NeuralPIL](https://github.com/cgtuebingen/Neural-PIL)
4. [PhySG](https://github.com/Kai-46/PhySG)
5. [PANDORA](https://github.com/akshatdave/pandora)


## Setting up

**1. Pull Mitsuba2 Docker image**

This code implementation requires [Mitsuba2](https://github.com/mitsuba-renderer/mitsuba2) version that has the `polarized plastic` BRDF implemented. The Docker image `akshatdave:mitsuba2_pplastic_image` contains such a Mitsuba2 distribution and can be downloaded through
```
docker pull akshatdave:mitsuba2_pplastic_image
```

**2. Create Mitsuba2 Docker container**

Edit the container name, port number and code folder in `helper_scripts/run_mitsuba_docker.sh` as required. 

To create a new container:
```
bash helper_scripts/run_mitsuba_docker.sh
```
Start the container 
```
docker container start <container name>
```
Run the container.
```
docker exec -it <container name> bash
```
The last line needs to be run on connecting to the container after first time. 

**3. Download example assets**

We provide example environment map and mesh to define the scene that can be downloaded from this [link](https://drive.google.com/file/d/1oZGMkND1VMokS3ZqMKK5xk-OrPykmbGN/view?usp=sharing). The file structure should be as
```
- misuba2_render_multiview/
    |- data/
        |- <environment map.exr>
        |- <object name>/
            |- mesh.obj
```
## Rendering

Create/edit the config file in `configs/` with required parameters. For example `bunny.txt`. Then run:
```
python3 scripts/01_render_multi_view_mitsuba.py --config configs/<config file>
```
Please refer to `config_parser` in the script `01_render_multi_view_mitsuba.py` for description of the parameters.

## Citation

This code base was developed as a part of our work PANDORA on multi-view inverse rendering using polarization cues and implicit neural representations
```
@article{dave2022pandora,
  title={PANDORA: Polarization-Aided Neural Decomposition Of Radiance},
  author={Dave, Akshat and Zhao, Yongyi and Veeraraghavan, Ashok},
  journal={arXiv preprint arXiv:2203.13458},
  year={2022}
}
```
