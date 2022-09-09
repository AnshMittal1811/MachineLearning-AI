# 2D-to-3D style transfer using Neural Renderer

This is the code for 2D-to-3D style transfer in the paper [Neural 3D Mesh Renderer (CVPR 2018)](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.

Related repositories:
* [Neural Renderer](https://github.com/hiroharu-kato/neural_renderer)
    * [Single-image 3D mesh reconstruction](https://github.com/hiroharu-kato/mesh_reconstruction)
    * [2D-to-3D style transfer](https://github.com/hiroharu-kato/style_transfer_3d)
    * [3D DeepDream](https://github.com/hiroharu-kato/deep_dream_3d)

## Installation
```
# install neural_renderer
git clone https://github.com/hiroharu-kato/neural_renderer.git
cd neural_renderer
python setup.py install --user
# or, sudo python setup.py install
```

## Example
```
bash ./examples/run.sh
```

<div>
   <img src="https://raw.githubusercontent.com/hiroharu-kato/style_transfer_3d/master/examples/data/styles/gris1.jpg" width="30%" height="30%">
   <img src="https://raw.githubusercontent.com/hiroharu-kato/style_transfer_3d/master/examples/data/results/teapot.gif" width="30%" height="30%">
   <img src="https://raw.githubusercontent.com/hiroharu-kato/style_transfer_3d/master/examples/data/results/teapot_gris.gif" width="30%" height="30%">
</div>

<div>
   <img src="https://raw.githubusercontent.com/hiroharu-kato/style_transfer_3d/master/examples/data/styles/munch1.jpg" width="30%" height="30%">
   <img src="https://raw.githubusercontent.com/hiroharu-kato/style_transfer_3d/master/examples/data/results/bunny.gif" width="30%" height="30%">
   <img src="https://raw.githubusercontent.com/hiroharu-kato/style_transfer_3d/master/examples/data/results/bunny_munch.gif" width="30%" height="30%">
</div>

## Citation

```
@inproceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```
