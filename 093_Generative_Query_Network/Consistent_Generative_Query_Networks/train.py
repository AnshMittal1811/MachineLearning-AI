import argparse, os, random

from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataset import GQNDataset, Scene, transform_viewpoint, sample_batch
from model import JUMP
from scheduler import AnnealingStepLR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network Implementation')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--dataset', type=str, default='Room', help='dataset (dafault: Room)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                        default="/workspace/dataset/GQN/rooms_free_camera_with_object_rotations-torch/train")
    parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                        default="/workspace/dataset/GQN/rooms_free_camera_with_object_rotations-torch/test")
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/workspace/logs')
    parser.add_argument('--log_dir', type=str, help='log directory (default: JUMP)', default='JUMP')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=100)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=10000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--layers', type=int, help='number of generative layers (default: 6)', default=6)
    parser.add_argument('--z_dim', type=int, default=3)
    parser.add_argument('--v_dim', type=int, default=5)
    parser.add_argument('--M', type=int, help='M in test', default=5)
    parser.add_argument('--num_epoch', type=int, help='number of epochs', default=40)
    parser.add_argument('--seed', type=int, help='random seed (default: None)', default=None)
    args = parser.parse_args()

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    
    # Seed
    if args.seed!=None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Dataset directory
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    # Number of workers to load data
    num_workers = args.workers

    # Log
    log_interval_num = args.log_interval
    save_interval_num = args.save_interval
    log_dir = os.path.join(args.root_log_dir, args.log_dir)
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'models'))
    os.mkdir(os.path.join(log_dir, 'optimizers'))
    
    os.mkdir(os.path.join(log_dir,'runs'))

    # TensorBoardX
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs'))

    # Dataset
    train_dataset = GQNDataset(root_dir=train_data_dir)
    test_dataset = GQNDataset(root_dir=test_data_dir)
    D = args.dataset

    # Number of scenes over which each weight update is computed
    B = args.batch_size
    
    M = args.M
    
    # Hyperparameters
    if D=='Narratives':
        nt=4
        stride_to_hidden=2
        nf_to_hidden=64
        nf_enc=128
        stride_to_obs=2
        nf_to_obs=128
        nf_dec=64
        nf_z=3
        nf_v=1
        alpha=2.0
        beta=0.5
    # elif D=='MNISTDice':
    else:
        nt=6
        stride_to_hidden=2
        nf_to_hidden=128
        nf_enc=128
        stride_to_obs=2
        nf_to_obs=128
        nf_dec=128
        nf_z=3
        nf_v=5
    # else:
        # raise NotImplementedError
        
    # Define model
    model = JUMP(nt, stride_to_hidden, nf_to_hidden, nf_enc, stride_to_obs, nf_to_obs, nf_dec, nf_z, nf_v).to(device)

    if len(args.device_ids)>1:
        model = nn.DataParallel(model, device_ids=args.device_ids)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = AnnealingStepLR(optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6)

    kwargs = {'num_workers':num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, **kwargs)

    f_data_test, v_data_test = next(iter(test_loader))
    N = f_data_test.size(1)

    step = 0
    # Training Iterations
    for epoch in range(args.num_epoch):
        for t, (f_data, v_data) in enumerate(tqdm(train_loader)):
            model.train()
            
            if D=='Narratives':
                pixel_var = max(beta + (alpha - beta)*(1 - step/(1e5)), beta)
            # elif D=='MNISTDice':
            else:
                if step < 1e5:
                    pixel_var = 2.0
                elif step < 1.5e5:
                    pixel_var = 0.2
                elif step < 2e5:
                    pixel_var = 0.4
                else:
                    pixel_var = 0.9
            # else:
                # raise NotImplementedError
                
            f, v = sample_batch(f_data, v_data, D)
            f, v = f.to(device), v.to(device)
            train_elbo, train_kl, train_mse = model(v, f, pixel_var)

            # Compute empirical ELBO gradients
            train_elbo.mean().backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Update optimizer state
            scheduler.step()

            # Logs
            writer.add_scalar('train_elbo', train_elbo.mean(), step)
            writer.add_scalar('train_kl', train_kl.mean(), step)
            writer.add_scalar('train_mse', train_mse.mean(), step)

            with torch.no_grad():
                model.eval()
                # Write logs to TensorBoard
                if step % log_interval_num == 0:
                    f_test, v_test = sample_batch(f_data_test, v_data_test, D, seed=0)
                    f_test, v_test = f_test.to(device), v_test.to(device)
                    
                    test_elbo, test_kl, test_mse = model(v_test, f_test, pixel_var)  

                    f_context, v_context = f_test[:, :M], v_test[:, :M]

                    if len(args.device_ids)>1:
                        f_prime = model.module.generate(v_test[:, :M], f_test[:, :M], v_test[:, M:])
                        f_hat = model.module.reconstruct(v_test, f_test)
                    else:
                        f_prime = model.generate(v_test[:, :M], f_test[:, :M], v_test[:, M:])
                        f_hat = model.reconstruct(v_test, f_test)
            
                    writer.add_scalar('test_elbo', test_elbo.mean(), step)
                    writer.add_scalar('test_kl', test_kl.mean(), step)
                    writer.add_scalar('test_mse', test_mse.mean(), step)
                    writer.add_image('test_context', make_grid(f_context.contiguous().view(-1, 3, 64, 64), M, pad_value=1), step)
                    writer.add_image('test_ground_truth', make_grid(f_test.contiguous().view(-1, 3, 64, 64), N, pad_value=1), step)
                    writer.add_image('test_generation', make_grid(f_prime.contiguous().view(-1, 3, 64, 64), N-M, pad_value=1), step)
                    writer.add_image('test_reconstruction', make_grid(f_hat.contiguous().view(-1, 3, 64, 64), N, pad_value=1), step)

            step += 1
            
        state_dict = model.module.state_dict() if len(args.device_ids)>1 else model.state_dict()
        torch.save(state_dict, log_dir + f"/models/model-{epoch}.pt")
        
        optimizer_state_dict = optimizer.state_dict()
        torch.save(optimizer_state_dict, log_dir + f"/optimizers/optimizer-{epoch}.pt")
                
    state_dict = model.module.state_dict() if len(args.device_ids)>1 else model.state_dict()
    torch.save(state_dict, log_dir + "/models/model-final.pt")  
    writer.close()
