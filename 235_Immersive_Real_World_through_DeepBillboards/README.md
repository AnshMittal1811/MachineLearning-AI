# DeepBillboard + PlenOctrees

With DeepBillboard, you can easily use your trained NeRF (or any other view synthesis models) in Unity without having to convert them to mesh! **DeepBillboard can also supports physical interaction**!

See the paper for more details.
https://sites.google.com/view/deepbillboards/

![samune](https://user-images.githubusercontent.com/23403885/183455399-63c42b5a-80f8-4730-957c-387db1027f2e.png)

## Easy 4 steps for use

1. Create Plane Object and attach `ControlBillboard.cs` and `texture.mat`.
1. Attach `ControlCamera.cs` to the main camera.
1. Run `python db_commands.py --input ./ckpt.npz`
1. Run Unity scene!

## related repositories

- https://github.com/naruya/svox
- https://github.com/naruya/plenoxel

Have a good DeepBillboard life!

## Acknowledgement
We'd like to express deep thanks to the inventors of [plenoctrees](https://github.com/sxyu/plenoctree),  [plenoxels](https://github.com/sxyu/svox2) and [nerf_pl](https://github.com/kwea123/nerf_pl).
