import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


def generate_trajectory(canvas=64, iters=2000, max_len=60, expl=0.005):
    """
    generate random trajectory as in [Boracchi and Foi 2012]
    :param canvas: kernel size
    :param iters: step number
    :param max_len:
    :param expl: anxiety probability
    :return: x[iters]
    """
    tot_length = 0
    big_expl_count = 0
    # how to be near the previous position
    # TODO: I can change this paramether for 0.1 and make kernel at all image
    centripetal = 0.7 * np.random.uniform(0, 1)
    # probability of big shake
    prob_big_shake = 0.2 * np.random.uniform(0, 1)
    # term determining, at each sample, the random component of the new direction
    gaussian_shake = 10 * np.random.uniform(0, 1)
    init_angle = 360 * np.random.uniform(0, 1)

    img_v0 = np.sin(np.deg2rad(init_angle))
    real_v0 = np.cos(np.deg2rad(init_angle))

    v0 = complex(real=real_v0, imag=img_v0)
    v = v0 * max_len / (iters - 1)

    if expl > 0:
        v = v0 * expl

    x = np.array([complex(real=0, imag=0)] * (iters))

    for t in range(0, iters - 1):
        if np.random.uniform() < prob_big_shake * expl:
            next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
            big_expl_count += 1
        else:
            next_direction = 0

        dv = next_direction + expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
                     max_len / (iters - 1))

        v += dv
        v = (v / float(np.abs(v))) * (max_len / float((iters - 1)))
        x[t + 1] = x[t] + v
        tot_length = tot_length + abs(x[t + 1] - x[t])

    # centere the motion
    x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
    x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
    x += complex(real=ceil((canvas - max(x.real)) / 2), imag=ceil((canvas - max(x.imag)) / 2))

    return x


def generate_PSF(trajectory, canvas=64):
    """
    motion kernels according to trajectory
    :param trajectory:
    :param canvas:
    :return:
    """
    canvas = (canvas, canvas)
    PSF = np.zeros(canvas)
    triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
    triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
    t_proportion = 1

    for t in range(len(trajectory)):
        m2 = int(np.minimum(canvas[1] - 1, np.maximum(1, np.math.floor(trajectory[t].real))))
        M2 = int(m2 + 1)
        m1 = int(np.minimum(canvas[0] - 1, np.maximum(1, np.math.floor(trajectory[t].imag))))
        M1 = int(m1 + 1)

        PSF[m1, m2] += t_proportion * triangle_fun_prod(
            trajectory[t].real - m2, trajectory[t].imag - m1
        )
        PSF[m1, M2] += t_proportion * triangle_fun_prod(
            trajectory[t].real - M2, trajectory[t].imag - m1
        )
        PSF[M1, m2] += t_proportion * triangle_fun_prod(
            trajectory[t].real - m2, trajectory[t].imag - M1
        )
        PSF[M1, M2] += t_proportion * triangle_fun_prod(
            trajectory[t].real - M2, trajectory[t].imag - M1
        )
    return PSF


def random_trajectory_kernels(batch, ksize=64, max_step=2000, anxiety=0.005):
    # batch_trajectory = np.zeros((batch, max_step))
    batch_psf = torch.FloatTensor(batch, ksize, ksize)
    for i in range(batch):
        x = generate_trajectory(canvas=ksize, iters=max_step, expl=anxiety)
        batch_psf[i] = torch.FloatTensor(generate_PSF(x, ksize))
    return batch_psf


def random_multi_frame_kernels(bsize, ksize, anxiety=1, pic_num=64):
    kernels = torch.zeros(bsize, 1, ksize, ksize)
    kernels[:, 0, ksize // 2, ksize // 2] = 1
    for i in range(pic_num):
        kernel_tmp = torch.zeros(bsize, 1, ksize, ksize)
        kernel_tmp[:, 0, ksize // 2, ksize // 2] = 1
        moved_center = (torch.rand(bsize, 2, 3) - 0.5) / 2
        moved_center[:, :, 0:2] = torch.eye(2).expand(bsize, 2, 2)
        moved_center = Variable(moved_center, volatile=True)
        affine = F.affine_grid(moved_center, kernel_tmp.size())
        kernel_tmp = F.grid_sample(kernel_tmp, affine)
        kernels += kernel_tmp.data

    kernels /= pic_num
    return kernels.data if isinstance(kernels, Variable) else kernels




























