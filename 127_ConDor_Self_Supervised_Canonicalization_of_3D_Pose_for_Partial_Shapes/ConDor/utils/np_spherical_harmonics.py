import numpy as np

# compute unnormalized spherical harmonics up to degree l_max <= 3 for a tensor of shape (..., 3)
def np_unnormalized_real_sh(l_max, r, cp, sp, ct, st):
    assert (3 >= l_max >= 1)

    phi = np.arctan2(cp, sp)
    # theta = np.arctan2(ct, st)


    Y00 = np.full(shape=x.shape, fill_value=np.sqrt(1. / np.pi) / 2.)

    Y0 = [Y00]

    # Y0 = tf.stack(Y0, axis=-1)

    cp = np.cos(phi)
    sp = np.sin(phi)

    Y1_10 = (np.sqrt(3. / (2.*np.pi)) / 2.) * r * cp*st
    Y1_11 = -(np.sqrt(3. / (2.*np.pi)) / 2.) * r * sp*st
    Y100 = (np.sqrt(3. / np.pi) / 2.) * ct
    # Y110 = - Y1_10
    # Y111 = Y1_11

    Y0 = [Y100] + Y0
    # Y1 = [Y1_10, Y1_11, Y1_10, Y111]
    # Y1 = [Y1_10, Y1_11]
    Y10 = [Y1_10]
    Y11 = [Y1_11]

    if l_max >= 2:

        c2p = np.cos(2. * phi)
        s2p = np.sin(2. * phi)
        st2 = st*st
        stct = st*ct
        ct2 = ct*ct
        r2 = r*r
        Y2_20 = (np.sqrt(15. / (2.*np.pi)) / 4.)*r2*c2p*st2
        Y2_21 = -(np.sqrt(15. / (2. * np.pi)) / 4.)*r2*s2p*st2

        Y2_10 = (np.sqrt(15. / (2. * np.pi)) / 2.)*r*cp*stct
        Y2_11 = -(np.sqrt(15. / (2. * np.pi)) / 2.)*r*st*stct

        Y200 = (np.sqrt(5. / np.pi) / 4.)*(3.*ct2 - 1.)

        Y0 = [Y200] + Y0
        Y10 = [Y2_10] + Y10
        Y11 = [Y2_11] + Y11
        Y20 = [Y2_20]
        Y21 = [Y2_21]

    if l_max >= 3:
        c3p = np.cos(3.* phi)
        s3p = np.sin(3.* phi)
        ct3 = ct2*ct
        st2ct = st2*ct
        st3 = st2*st
        stct2 = st*ct2
        r3 = r2*r
        Y3_30 = (np.sqrt(35. / np.pi) / 8.) * r3 *c3p * st3
        Y3_31 = -(np.sqrt(35. / np.pi) / 8.) * r3 * s3p * st3

        Y3_20 = (np.sqrt(35. / (2. * np.pi)) / 4.) * r2 *c2p * st2ct
        Y3_21 = -(np.sqrt(35. / (2. * np.pi)) / 4.) * r2 *s2p * st2ct

        Y3_10 = (np.sqrt(21. / np.pi) / 8.) * r * cp*(5.*stct2 - st)
        Y3_11 = -(np.sqrt(21. / np.pi) / 8.) * r * sp*(5.*stct2 - st)

        Y300 = (np.sqrt(7. / np.pi) / 4.)*(5*ct3 - 3.*ct)

        Y0 = [Y300] + Y0
        Y10 = [Y3_10] + Y10
        Y11 = [Y3_11] + Y11
        Y20 = [Y3_20] + Y20
        Y21 = [Y3_21] + Y21
        Y30 = [Y3_30]
        Y31 = [Y3_31]


    if l_max == 1:
        Y_0 = [Y10]
        Y_1 = [Y11]
    elif l_max == 2:
        Y_0 = [Y10, Y20]
        Y_1 = [Y11, Y21]
    else:
        Y_0 = [Y10, Y20, Y30]
        Y_1 = [Y11, Y21, Y31]

    for i in range(len(Y_0)):
        Y_0[i] = np.stack(Y_0[i], axis=-1)
        Y_1[i] = np.stack(Y_1[i], axis=-1)

    return Y0, Y_0, Y_1
