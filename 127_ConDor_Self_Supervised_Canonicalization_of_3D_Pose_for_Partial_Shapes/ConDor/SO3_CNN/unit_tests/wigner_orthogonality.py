import tensorflow as tf
import numpy as np
from SO3_CNN.tf_wigner import *
from SO3_CNN.sampling import tf_SO3_fps

def test_coeffs(l):
    c = np.zeros(((2*l+1)**2, 2*l+1, 2*l+1, 1))
    for i in range(2*l+1):
        for j in range(2*l+1):
            c[i*(2*l+1) + j, i, j, 0] = 1.0
    return tf.convert_to_tensor(c, dtype=tf.float32)

"""
l_max = 2

s = tf_SO3_fps(num_samples=1024, res=20)

coeffs = WignerCoeffs(base=s, l_list=[l_max])
eval = WignerEval(base=s, l_list=[l_max])

c = test_coeffs(l_max)

D = coeffs.D
D = tf.reshape(D, (D.shape[0], 2*l_max+1, 2*l_max+1))

e = eval.compute({str(l_max) : c})

c_ = coeffs.compute(e)
u = c_[str(l_max)]
print(c - tf.where(u < 0.03, 0., u))

"""


"""
e = e[..., 0]
e = tf.transpose(e, (1, 0))
e = tf.reshape(e, (-1, 3, 3))
"""



"""
print(D.shape)


u = tf.einsum('ijk,iab->ijkab', D, D)
u = 3 * tf.reduce_sum(u, axis=0, keepdims=False) / (u.shape[0])

u = tf.where(u < 0.03, 0., u)
print(u)
"""

"""
coeffs = WignerCoeffs(base=s, l_list=[l_max])
eval = WignerEval(base=s, l_list=[l_max])

c_ = test_coeffs(l_max)
e = eval.compute({str(l_max) : c_})
c = coeffs.compute(e)

print(c)
print(c_)


print(c[str(l_max)] - c_)
"""

def test_coeffs_2(c, idx):
    y = dict()
    for l in c:
        cl = np.zeros((1, 2*l+1, 2*l+1, 1))
        cl[0, idx[l][0], idx[l][1], 0] = c[l]
        y[str(l)] = tf.convert_to_tensor(cl, dtype=tf.float32)
    return y

c = {0: 0.5, 1: 1., 2: 0.7}
idx = {0: [0, 0], 1: [2, 1], 2: [2, 0]}
c = test_coeffs_2(c, idx)

L = []
for l in c:
    L.append(int(l))




s = tf_SO3_fps(num_samples=1024, res=20)
coeffs = WignerCoeffs(base=s, l_list=L)
eval = WignerEval(base=s, l_list=L)

e = eval.compute(c)
c_ = coeffs.compute(e)

for l in c_:
    c_l = c_[l]
    c_[l] = tf.where(c_l < 0.03, 0., c_l)

print(c_)