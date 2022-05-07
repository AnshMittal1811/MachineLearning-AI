import numpy as np
from SO3_CNN.tf_wigner import tf_wigner_matrix, zyz_euler_angles_rot, zyz_euler_angles, matrix_to_euler_zyz_angles
from spherical_harmonics.wigner_matrix import real_D_wigner
import tensorflow as tf
from utils.pointclouds_utils import setup_pcl_viewer
import vispy
from SO3_CNN.sampling import S2_fps, SO3_fps, rot_z, rot_y, SO3_sampling_from_S2, tf_polyhedrons
from SO3_CNN.spherical_harmonics_ import tf_spherical_harmonics
from utils.pointclouds_utils import generate_3d
from spherical_harmonics.wigner_matrix import euler_rot_zyz

"""
wigner numpy vs tensorflow
"""

"""
l_max = 4
a = np.random.rand(3)
# a[0] = 0
# a[1] = 0
# a[2] = 0
# a = np.array([0.4, 0.0, 0.0])
tf_a = tf.convert_to_tensor(a, dtype=tf.float32)
euler = matrix_to_euler_zyz_angles(zyz_euler_angles_rot(tf_a[..., 0], tf_a[..., 1], tf_a[..., 2]))


print('euler angles')
a_ = a
# a_[1] /= 2
print(np.cos(a_))
print(np.sin(a_))
print('euler from mat')
print(euler)


w_test = dict()
for l in range(l_max+1):
    w_test[str(l)] = real_D_wigner(l, a[0], a[1], a[2])

wigner = tf_wigner_matrix(l_max=l_max)
# w = wigner.compute(zyz_euler_angles_rot(tf_a[..., 0], tf_a[..., 1], tf_a[..., 2]))
w = wigner.compute_euler_zyz(tf_a)
for l in range(l_max+1):

    # print('old')
    # print(w_test[str(l)])
    # print('new')
    # print(w[str(l)])



    print("diff")
    print(tf.norm(w[str(l)] - w_test[str(l)], axis=[-2, -1]) / tf.norm(w_test[str(l)], axis=[-2, -1]))


def wigner_basis_size(n):
    return 4 * n*(n+1)*(2*n+1) / 6 + 2*n*(n+1) + n+1

print(wigner_basis_size(6))
"""
"""
'tetrakis_hexahedron':tetrakis_hexahedron
         'regular_dodecahedron':regular_dodecahedron,
         'pentakis_dodecahedron':pentakis_dodecahedron,
         'disdyakis_triacontahedron':disdyakis_triacontahedron
"""

"""
Samplings
"""


"""
# R = SO3_fps(num_samples=256, res=40)
# S = S2_fps(num_samples=32, res=100)
S = polyhedrons('pentakis_dodecahedron')
S = tf.convert_to_tensor(S, dtype=tf.float32)
R = SO3_sampling_from_S2(S, k=16)
R = np.array(R)
print('uuu', R.shape)
# R1 = SO3_fps(num_samples=1024, res=34)
# R = R - R1
# R = SO3_fps(num_samples=1024, res=35)

n = np.array([0, 0, 1])
y = np.einsum('i,...ij->...j', n, R)

setup_pcl_viewer(X=R[..., 1], color=(1, 1, 1, .5), run=True)
vispy.app.run()

S = np.array(S)
setup_pcl_viewer(X=S, color=(1, 1, 1, .5), run=True)
vispy.app.run()
"""


"""
tf wigner equivariance
"""

"""
l_max = 6
x = np.random.rand(3)
x = x / np.linalg.norm(x)
# R = generate_3d()



a = np.random.rand(3)

a[0] *= 1.
a[1] *= 1.
a[2] *= 1.

R = euler_rot_zyz(a[0], a[1], a[2])

# R = rot_z(np.array([0.2]))
# R = R[0]
Rx = np.einsum('ij,...j->...i', R, x)

Y = tf_spherical_harmonics(l_max=l_max)
D = tf_wigner_matrix(l_max=l_max)

R = tf.convert_to_tensor(R, dtype=tf.float32)
w_test = dict()
for l in range(l_max+1):
    w_test[str(l)] = real_D_wigner(l, a[0], a[1], a[2])

a = tf.convert_to_tensor(a, dtype=tf.float32)
x = tf.convert_to_tensor(x, dtype=tf.float32)
Rx = tf.convert_to_tensor(Rx, dtype=tf.float32)






# d = D.compute_euler_zyz(a)
d = D.compute(R)
y = Y.compute(x)
Ry = Y.compute(Rx)


for l in range(l_max+1):



    dy = tf.einsum('...ij,...j->...i', d[str(l)], y[str(l)])
    print(tf.norm(Ry[str(l)] - dy))

print("R")
print(R)
print("w")
print(d[str(1)])
print("x")
print(x / np.linalg.norm(x))
print("y")
print(y[str(1)] / tf.linalg.norm(y[str(1)]))
"""