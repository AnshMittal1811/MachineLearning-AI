# Chamfer Distance API
Chamfer Distance (CD) is a evaluation metric for two point clouds. It takes the distance of each points into account. For each point in each cloud, CD finds the nearest point in the other point set, and sums the square of distance up. It is utilized in Shapenet’s shape reconstruction challenge.

The chamfer distance between point cloud *S1* and *S2* is defined as 

![](https://github.com/UM-ARM-Lab/Chamfer-Distance-API/blob/master/formula%20for%20chamfer%20distance.png)

This readme is a guidance on how to compile the code for chamfer distance and a tutorial to use chamfer distance api.

## Prerequisites
- [`GCC`](https://gcc.gnu.org) 5.4.0
- [`CUDA`](https://developer.nvidia.com/cuda-toolkit) 9.0
- [`Python`](https://www.python.org) 2.7.12
- [`TensorFlow`](https://github.com/tensorflow/tensorflow) 1.7.0


## Compiling the chamfer-distance code
The folder `chamfer-distance` contains tensorflow module for chamfer-distance. To compile them, make sure tensorflow is installed. Then, modify the first 4 lines of `chamfer-distance/Makefile` according to your environment. Finally, compile the chamfer-distance code 
```sh
# From chamfer-distance/
make
```

To test the code, try 
```sh
# From chamfer-distance/
python tf_nndistance.py
```

## Chamfer-Distance-API
After compiling code for tensorflow, we can use the Chamfer Distance API now. Note the chamfer distance is defined as the sum of **square** of euclidean distance in the API. Try  
```sh
python chamfer_test.py
```
which will calculate chamfer distance between point (1, 1, 1) and (1, 1, -1) and output [8.].

Generally, to use the api, use `cd_api = Chamfer_distance()` to initialize the class, and then use `cd_api.get_chamfer_distance(xyz1,xyz2)` to calculate the chamfer distance. 



## References
* The .cu and .cpp files are from [PointSetGeneration](https://github.com/fanhqme/PointSetGeneration)
* The .py file is modified from [shapenet iccv 2017](https://shapenet.cs.stanford.edu/iccv17/)
* Makefile is modified from [pix3d](https://github.com/xingyuansun/pix3d) and [tensorflow: build_the_op_library](https://www.tensorflow.org/extend/adding_an_op#build_the_op_library)




