import tensorflow as tf
from SO3_CNN.so3_conv import SO3Conv
from SO3_CNN.spherical_harmonics_ import SphericalHarmonicsCoeffs, SphericalHarmonicsEval
from SO3_CNN.tf_wigner import WignerEval, WignerCoeffs, norms, S2_to_SO3_pullback
from SO3_CNN.sampling import tf_S2_fps, SO3_sampling_from_S2, SO3_fps, tf_polyhedrons, tf_SO3_fps
from SO3_CNN.sphere_embeddings import sphere_shell_embedding
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation, Dense, Dropout


class SphericalCNNCls(tf.keras.Model):
    def __init__(self, num_classes):
        super(SphericalCNNCls, self).__init__()
        self.num_classes = num_classes
        self.droupout_rate = 0.5
        self.bn_momentum = 0.75
        # self.dodecahedron = 'pentakis'
        # self.dodecahedron = 'regular'
        # self.d = 3
        self.l_max = [5, 4, 3]
        self.num_shells = 3

        # self.shell_radius = [0.33, 0.66, 1.0]
        # self.num_shell_samples = [64, 256, 512]

        self.shell_radius = [0.4, 0.8, 1.2]
        self.num_shell_samples = [512, 512, 512]

        self.sphere_base = ['pentakis_dodecahedron', 'pentakis_dodecahedron', 'pentakis_dodecahedron']

        # self.sphere_base = ['regular_dodecahedron', 'regular_dodecahedron', 'regular_dodecahedron']

        self.num_local_dirs = [32, 32, 32]
        self.S2_base = []
        self.SO3 = []
        self.wigner_eval = []
        self.wigner_coeffs = []
        self.so3_conv = []
        self.units = [16, 32, 64]

        self.sph_coeffs = []

        for i in range(len(self.shell_radius)):
            s = tf_S2_fps(num_samples=self.num_shell_samples[i])
            self.sph_coeffs.append(SphericalHarmonicsCoeffs(base=s, l_max=self.l_max[0]))

        self.bn_layers = []
        for i in range(len(self.sphere_base)):
            self.bn_layers.append(BatchNormalization(momentum=self.bn_momentum))


            p = tf_polyhedrons(self.sphere_base[i])
            self.S2_base.append(p)
            s = SO3_sampling_from_S2(p, self.num_local_dirs[i])


            # s = tf_SO3_fps(num_samples=512, res=20)


            self.SO3.append(s)
            self.so3_conv.append(SO3Conv(units=self.units[i], name_=str(i)))
            self.wigner_eval.append(WignerEval(base=s, l_max=self.l_max[i]))
            self.wigner_coeffs.append(WignerCoeffs(base=s, l_max=self.l_max[i]))

        self.fc1_units = 512
        self.fc2_units = 256

        self.fc1 = Dense(units=self.fc1_units, activation=None)
        # if with_bn:
        self.bn_fc1 = BatchNormalization(momentum=self.bn_momentum)
        # self.activation1 = Activation('relu')
        # self.drop1 = Dropout(rate=self.droupout_rate)
        self.fc2 = Dense(units=self.fc2_units, activation=None)
        # if with_bn:
        self.bn_fc2 = BatchNormalization(momentum=self.bn_momentum)
        # self.activation1 = Activation('relu')
        # self.drop2 = Dropout(rate=self.droupout_rate)
        self.softmax = Dense(units=self.num_classes, activation='softmax')

    def call(self, x):

        y = sphere_shell_embedding(x, coeffs=self.sph_coeffs, r=self.shell_radius)


        y = S2_to_SO3_pullback(y)



        for i in range(len(self.so3_conv)):
            y = self.so3_conv[i](y)

            y = self.wigner_eval[i].compute(y)

            y = self.bn_layers[i](y)
            y = LeakyReLU()(y)
            if i < len(self.so3_conv) - 1:
                y = self.wigner_coeffs[i+1].compute(y)





        y = self.wigner_coeffs[-1].compute(y)
        # y = norms(y, axis=-3)

        # y = norms(y, axis=-3)
        # y = self.wigner_eval[0].compute(y)
        # y = self.wigner_coeffs[-1].compute(y)
        l = 1
        # y = {'0': y['0'], '1': y['1'], '2': y['2']}

        """
        e = WignerEval(base=self.SO3[0], l_list=[0, 1, 2])
        y = e.compute(y)
        c = WignerCoeffs(base=self.SO3[0], l_list=[0, 1, 2])
        y = c.compute(y)
        """

        y = norms(y, axis=-3)

        # y = tf.reduce_max(y, axis=1, keepdims=False)

        # y = self.coeffs[-1](y)
        # y = norms(y)

        print('last y shape ', y.shape)
        y = self.fc1(y)
        y = self.bn_fc1(y)
        y = Activation('relu')(y)
        y = Dropout(rate=self.droupout_rate)(y)
        y = self.fc2(y)
        y = self.bn_fc2(y)
        y = Activation('relu')(y)
        y = Dropout(rate=self.droupout_rate)(y)
        y = self.softmax(y)
        return y