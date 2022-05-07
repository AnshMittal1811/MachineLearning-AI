from SO3_CNN.sampling import S2_fps, SO3_fps, SO3_sampling_from_S2, tf_polyhedrons, tf_sq_distance_matrix, np_polyhedrons, tf_S2_fps
from SO3_CNN.spherical_harmonics_ import tf_spherical_harmonics
import vispy
from utils.pointclouds_utils import setup_pcl_viewer
import tensorflow as tf
import numpy as np
from SO3_CNN.unimodality_regularizers import S2UnimodalityRegularizer
from SO3_CNN.spherical_harmonics_ import SphericalHarmonicsEval, SphericalHarmonicsCoeffs


class S2Init:
    def __init__(self, points_idx, values, k, l_max=3, base='pentakis'):
        super(S2Init, self).__init__()
        assert(points_idx.shape[0] == values.shape[0])
        self.points_idx = tf.convert_to_tensor(points_idx, tf.int32)
        # self.values = tf.convert_to_tensor(values, tf.float32)
        self.P = tf_polyhedrons(base)
        self.points = tf.gather(self.P, self.points_idx, axis=0)
        self.k = k


        sq_dist_mat = tf_sq_distance_matrix(self.P, self.points)
        _, patches_idx = tf.nn.top_k(-sq_dist_mat, k=self.k)
        # patches_idx = tf.reshape(patches_idx, (-1,))
        patches_idx = np.array(patches_idx, dtype=np.int32)

        v = np.zeros((self.P.shape[0], 1))
        for i in range(self.points.shape[0]):
            for j in range(self.k):
                v[patches_idx[i, j], 0] += values[i]

        self.values = tf.convert_to_tensor(v, dtype=tf.float32)

        self.C = SphericalHarmonicsCoeffs(l_max=l_max, base=base)
        self.init_coeffs = self.C.compute(self.values)

        self.coeffs = dict()
        for l in self.init_coeffs:
            self.coeffs[l] = tf.Variable(initial_value=self.init_coeffs[l], trainable=True)

    def get(self):
        return self.coeffs

def color(f, alpha=0.8):
    pos = np.array([0., 1., 0., 0.8])
    neg = np.array([0., 0., 1., 0.8])

    c = np.zeros((f.shape[0], 4), dtype=np.float32)

    max_ = -10000000.
    min_ = +10000000.
    for i in range(f.shape[0]):
        max_ = max(max_, f[i])
        min_ = min(min_, f[i])


    for i in range(f.shape[0]):
        v = (f[i, 0] - min_) / max((max_ - min_), 0.0001)
        c[i, 3] = alpha
        for j in range(3):
            c[i, j] = v*pos[j] + (1-v)*neg[j]

    return c


points_idx = np.array([0, 47], dtype=np.int32)
values = np.array([1., 0.7], dtype=np.float32)

l_max = 5

coeffs = S2Init(points_idx=points_idx,
           values=values,
           k=9, l_max=l_max, base='disdyakis_triacontahedron').get()

S2_samples = tf_S2_fps(4096, res=100)
init_eval = SphericalHarmonicsEval(l_max=l_max, base=S2_samples).compute(coeffs)
init_eval = np.array(init_eval, dtype=np.float32)


# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=10.*1e-3)
# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# regularizer = S2UnimodalityRegularizer(num_samples=256, l_max=l_max, regularizer='max_dirac')

# regularizer = S2UnimodalityRegularizer(num_samples=256, l_max=l_max, regularizer='max_dirac')

regularizer = S2UnimodalityRegularizer(num_samples=256, l_max=l_max, regularizer='uu', num_neighbours=5)

nb_iter = 300

print(coeffs)
for iter in range(nb_iter):
    print("iter", iter)
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        loss_value = regularizer.loss(coeffs)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, list(coeffs.values()))

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.

        optimizer.apply_gradients(zip(grads, list(coeffs.values())))
        print('loss ', float(loss_value))




setup_pcl_viewer(X=np.array(S2_samples, dtype=np.float32),
                 color=color(init_eval, alpha=1.0),
                 run=True, point_size=15)
vispy.app.run()

opt_eval = SphericalHarmonicsEval(l_max=l_max, base=S2_samples).compute(coeffs)
setup_pcl_viewer(X=np.array(S2_samples, dtype=np.float32),
                 color=color(np.array(opt_eval, dtype=np.float32), alpha=1.0),
                 run=True, point_size=15)
vispy.app.run()

