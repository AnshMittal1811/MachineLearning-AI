nvcc := /usr/local/cuda-9.0/bin/nvcc
cudalib := /usr/local/cuda-9.0/lib64
cudainclude := /usr/local/cuda-9.0/include
tensorflow := /home/armlab/catkin_thomas_ws/src/Shape-Completion/3D-RecGAN++/env/lib/python2.7/site-packages/tensorflow/include

TF_CFLAGS := $$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') 
TF_LFLAGS := $$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') 


all: tf_nndistance_so.so
clean:
	rm -rf *.o *.so
.PHONY : all clean

tf_nndistance_so.so: tf_nndistance_g.o tf_nndistance.cpp
	g++ -std=c++11 -shared tf_nndistance.cpp tf_nndistance_g.o -o tf_nndistance_so.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2

tf_nndistance_g.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

