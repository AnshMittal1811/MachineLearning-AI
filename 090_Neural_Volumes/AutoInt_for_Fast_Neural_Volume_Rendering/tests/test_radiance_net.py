import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modules
import torch
from diff_operators import jacobian, gradient
import numpy as np
torch.backends.cudnn.benchmark = True


def run_forward():

    sampler = modules.SamplingNet(Nt=128, ncuts=32)
    model = modules.RadianceNet(input_processing_fn=modules.input_processing_fn, sampler=sampler, use_grad=True)
    model.cuda()

    t = torch.rand(10, 128, 1).cuda()
    ray_dirs = torch.rand(10, 1, 3).cuda()
    ray_origins = torch.rand(10, 1, 3).cuda()
    orientations = torch.rand(10, 1, 6).cuda()
    model_in = {'t': t, 'ray_directions': ray_dirs, 'ray_origins': ray_origins, 'ray_orientations': orientations}

    print('Testing grad network timings...')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up
    model(model_in)
    times = []
    for i in range(100):
        model.set_mode('grad')
        start.record()
        out = model(model_in)['model_out']['output']
        loss = torch.sum(out)
        loss.backward()
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print(f'Average time for forward + backward: {np.mean(times):.02f} ms')


def check_backward():
    sampler = None
    input_processing_fn = None
    sigma_model = modules.RadianceNet(input_processing_fn=input_processing_fn, sampler=sampler, input_name=['ray_samples'])
    rgb_model = modules.RadianceNet(input_processing_fn=input_processing_fn, sampler=sampler)

    for name, model in {'sigma': sigma_model, 'rgb': rgb_model}.items():
        print(f'Checking gradients for {name} model')

        t = torch.rand(128, 64, 1).cuda()
        ray_dirs = torch.rand(128, 64, 3).cuda()
        ray_origins = torch.rand(128, 64, 3).cuda()
        orientations = torch.rand(128, 64, 6).cuda()

        model_in = {'t': t, 'ray_directions': ray_dirs, 'ray_origins': ray_origins, 'ray_orientations': orientations}
        our_grad = model(model_in)['model_out']['output'].squeeze()

        t = torch.nn.Parameter(t)
        model.set_mode('integral')
        model_in = {'t': t, 'ray_directions': ray_dirs, 'ray_origins': ray_origins, 'ray_orientations': orientations}
        model_out = model(model_in)
        out = model_out['model_out']['output']

        pytorch_grad = jacobian(out, model_out['model_in']['t'])[0].squeeze()

        #print(torch.abs(our_grad - pytorch_grad).max())
        print('Passed' if torch.allclose(our_grad, pytorch_grad, atol=1e-6) else 'Failed!')


if __name__ == '__main__':
    run_forward()
    check_backward()
